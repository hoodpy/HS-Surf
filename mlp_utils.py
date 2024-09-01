import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np
import utils

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")

class Hash_Encoder(nn.Module):
	def __init__(self, box_min, box_max, base, finest, scale, units_nerf=64, units_ngp=2, levels=16, hash_size=20):
		super(Hash_Encoder, self).__init__()
		self.box_min = torch.tensor(box_min, dtype=torch.float32, requires_grad=False, device=device)
		self.box_max = torch.tensor(box_max, dtype=torch.float32, requires_grad=False, device=device)
		self.base = float(base)
		self.finest = float(finest) if finest is not None else None
		self.scale = float(scale) if scale is not None else None
		self.units_nerf = units_nerf
		self.units_ngp = units_ngp
		self.levels = levels
		self.hash_size = hash_size
		self.gemo_grid = self.build_hash_grid()
		self.color_grid = self.build_hash_grid()

	def build_hash_grid(self):
		if self.finest is not None:
			ratio = np.exp((np.log(self.finest) - np.log(self.base)) / (self.levels - 1))
		elif self.scale is not None:
			ratio = self.scale
		else:
			raise ValueError("finest and scale can not be same as None, priority: finest > scale")

		encoding_config = {"otype": "HashGrid", "n_levels": self.levels, "n_features_per_level": self.units_ngp, 
			"log2_hashmap_size": self.hash_size, "base_resolution": int(self.base), "per_level_scale": ratio}

		network_config = {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "ReLU", 
			"n_neurons": self.units_nerf, "n_hidden_layers": 1}

		embeddings = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=self.units_nerf, 
			encoding_config=encoding_config, network_config=network_config).to(device)

		return embeddings

	def forward(self, pos_xyz):
		pos_xyz = torch.reshape(pos_xyz, [-1, 3])

		mask = torch.eq(torch.maximum(torch.minimum(pos_xyz, self.box_max), self.box_min), pos_xyz)
		mask = torch.eq(torch.sum(mask.to(dtype=torch.int32), dim=-1), 3).to(dtype=torch.float32)

		pos_xyz = torch.maximum(torch.minimum(pos_xyz, self.box_max), self.box_min)
		locations = torch.div(torch.sub(pos_xyz, self.box_min), torch.sub(self.box_max, self.box_min))

		gemo_encodings = self.gemo_grid(locations)
		color_encodings = self.color_grid(locations)

		gemo_encodings = gemo_encodings.to(dtype=torch.float32)
		color_encodings = color_encodings.to(dtype=torch.float32)

		return gemo_encodings, color_encodings, mask


class Get_Inters_Property(nn.Module):
	def __init__(self, num_rays, pos_levels, dir_levels, units_nerf, app_dims, appearance=True):
		super(Get_Inters_Property, self).__init__()
		self.num_rays = num_rays
		self.pos_levels = pos_levels
		self.dir_levels = dir_levels
		self.units_nerf = units_nerf
		self.app_dims = app_dims
		self.appearance = appearance
		self.build()

	def build(self):
		pos_dims = self.pos_levels * 6 + 3
		dir_dims = self.dir_levels * 6 + 3

		self.dense0 = nn.Sequential(nn.Linear(pos_dims, self.units_nerf, bias=True), nn.ReLU()).to(device)
		self.dense1 = nn.Sequential(nn.Linear(self.units_nerf, self.units_nerf, bias=True), nn.ReLU()).to(device)

		self.dense2 = nn.Sequential(nn.Linear(self.units_nerf*2, self.units_nerf, bias=True), nn.ReLU()).to(device)
		self.sigmas = nn.Sequential(nn.Linear(self.units_nerf, 1, bias=True), nn.Softplus()).to(device)

		self.dense3 = nn.Sequential(nn.Linear(self.units_nerf*2, self.units_nerf, bias=True), nn.ReLU()).to(device)

		if self.appearance:
			self.dense4 = nn.Sequential(nn.Linear(dir_dims+self.units_nerf+self.app_dims, self.units_nerf, bias=True), 
				nn.ReLU()).to(device)
		else:
			self.dense4 = nn.Sequential(nn.Linear(dir_dims+self.units_nerf, self.units_nerf, bias=True), nn.ReLU()).to(device)

		self.rgbs = nn.Sequential(nn.Linear(self.units_nerf, 3, bias=True), nn.Sigmoid()).to(device)

	def forward(self, pos_encodings, dir_encodings, app_encodings, gemo_encodings, color_encodings, mask, num_inters):
		if self.appearance: app_encodings = torch.tile(app_encodings, [self.num_rays * num_inters, 1])

		dense0 = self.dense0(pos_encodings)
		dense1 = self.dense1(dense0)

		dense2 = self.dense2(torch.cat([dense1, gemo_encodings], dim=-1))
		sigmas = self.sigmas(dense2)
		sigmas = torch.mul(torch.squeeze(sigmas, dim=-1), mask)

		dense3 = self.dense3(torch.cat([dense1, color_encodings], dim=-1))

		if self.appearance:
			dense4 = self.dense4(torch.cat([dense3, dir_encodings, app_encodings], dim=-1))
		else:
			dense4 = self.dense4(torch.cat([dense3, dir_encodings], dim=-1))

		rgbs = self.rgbs(dense4)

		sigmas = torch.reshape(sigmas, [self.num_rays, num_inters])
		rgbs = torch.reshape(rgbs, [self.num_rays, num_inters, 3])

		return sigmas, rgbs


class Depth_Correct(nn.Module):
	def __init__(self, box_center, poi_levels, dir_levels, units_dec):
		super(Depth_Correct, self).__init__()
		self.box_center = torch.tensor(box_center, dtype=torch.float32, device=device)
		self.poi_levels = poi_levels
		self.dir_levels = dir_levels
		self.units_dec = units_dec
		self.build()

	def build(self):
		poi_dims = self.poi_levels * 6 + 3
		dir_dims = self.dir_levels * 6 + 3
		self.poi_encoder = utils.Point_Encoder(self.poi_levels)

		self.dense0 = nn.Sequential(nn.Linear(poi_dims, self.units_dec, bias=True), nn.ReLU()).to(device)
		self.dense1 = nn.Sequential(nn.Linear(self.units_dec, self.units_dec, bias=True), nn.ReLU()).to(device)
		self.dense2 = nn.Sequential(nn.Linear(self.units_dec, self.units_dec, bias=True), nn.ReLU()).to(device)
		self.dense3 = nn.Sequential(nn.Linear(self.units_dec, self.units_dec, bias=True), nn.ReLU()).to(device)

		self.dense4 = nn.Sequential(nn.Linear(self.units_dec+dir_dims, self.units_dec//2, bias=True), nn.ReLU()).to(device)
		self.dense5 = nn.Linear(self.units_dec//2, 1, bias=True, device=device)

	def forward(self, poi_xyz, dir_encodings, depths_c2):
		poi_xyz = torch.sub(poi_xyz, self.box_center)
		poi_encodings = self.poi_encoder(poi_xyz)

		poi_encodings = poi_encodings.detach()
		depths_c2 = depths_c2.detach()

		dense0 = self.dense0(poi_encodings)
		dense1 = self.dense1(dense0)
		dense2 = self.dense2(dense1)
		dense3 = self.dense3(dense2)
		dense4 = self.dense4(torch.cat([dense3, dir_encodings], dim=-1))
		dense5 = self.dense5(dense4)

		results = torch.add(dense5, torch.unsqueeze(depths_c2, dim=-1))

		return results


class Get_Range(nn.Module):
	def __init__(self, units_ger, min_val=1./2000., max_val=1./50.):
		super(Get_Range, self).__init__()
		self.units_ger = units_ger
		self.min_val = min_val
		self.max_val = max_val
		self.build()

	def build(self):
		self.dense0 = nn.Sequential(nn.Linear(2, self.units_ger, bias=True), nn.ReLU()).to(device)
		self.dense1 = nn.Sequential(nn.Linear(self.units_ger, self.units_ger, bias=True), nn.ReLU()).to(device)
		self.dense2 = nn.Sequential(nn.Linear(self.units_ger, self.units_ger, bias=True), nn.ReLU()).to(device)
		self.dense3 = nn.Sequential(nn.Linear(self.units_ger, 2, bias=True), nn.ReLU()).to(device)

	def forward(self, t_near, t_far, s_vals):
		inputs = torch.cat([t_near, t_far], dim=-1)

		dense0 = self.dense0(inputs)
		dense1 = self.dense1(dense0)
		dense2 = self.dense2(dense1)
		dense3 = self.dense3(dense2)

		k = torch.unsqueeze(dense3[:, 0], dim=-1)
		b = torch.unsqueeze(dense3[:, 1], dim=-1)

		s_range = torch.mul(torch.add(torch.mul(k, s_vals), b), 0.1)
		s_range = torch.clamp(s_range, min=self.min_val, max=self.max_val)

		t_range = torch.mul(torch.sub(t_far, t_near), s_range)

		return t_range


class Get_Fine_Property(nn.Module):
	def __init__(self, num_rays, pos_levels, dir_levels, units_mlp, app_dims, appearance=True):
		super(Get_Fine_Property, self).__init__()
		self.num_rays = num_rays
		self.pos_levels = pos_levels
		self.dir_levels = dir_levels
		self.units_mlp = units_mlp
		self.app_dims = app_dims
		self.appearance = appearance
		self.build()

	def build(self):
		pos_dims = self.pos_levels * 6 + 3
		dir_dims = self.dir_levels * 6 + 3
		self.pos_encoder = utils.Position_Encoder(self.pos_levels)

		self.dense0 = nn.Sequential(nn.Linear(pos_dims, self.units_mlp, bias=True), nn.ReLU()).to(device)
		self.dense1 = nn.Sequential(nn.Linear(self.units_mlp, self.units_mlp, bias=True), nn.ReLU()).to(device)
		self.dense2 = nn.Sequential(nn.Linear(self.units_mlp, self.units_mlp, bias=True), nn.ReLU()).to(device)
		self.dense3 = nn.Sequential(nn.Linear(self.units_mlp, self.units_mlp, bias=True), nn.ReLU()).to(device)

		self.dense4 = nn.Sequential(nn.Linear(pos_dims+self.units_mlp, self.units_mlp), nn.ReLU()).to(device)
		self.dense5 = nn.Sequential(nn.Linear(self.units_mlp, self.units_mlp, bias=True), nn.ReLU()).to(device)
		self.dense6 = nn.Sequential(nn.Linear(self.units_mlp, self.units_mlp, bias=True), nn.ReLU()).to(device)
		self.dense7 = nn.Sequential(nn.Linear(self.units_mlp, self.units_mlp, bias=True), nn.ReLU()).to(device)

		if self.appearance:
			self.dense8 = nn.Sequential(nn.Linear(self.units_mlp+dir_dims+self.app_dims, self.units_mlp//2), 
				nn.ReLU()).to(device)
		else:
			self.dense8 = nn.Sequential(nn.Linear(self.units_mlp+dir_dims, self.units_mlp//2), nn.ReLU()).to(device)

		self.dense9 = nn.Linear(self.units_mlp//2, 3, bias=True, device=device)

	def forward(self, means, covs, dir_encodings, app_encodings, colors_c2):
		if self.appearance: app_encodings = torch.tile(app_encodings, [self.num_rays, 1])

		pos_encodings = self.pos_encoder(means, covs)

		dense0 = self.dense0(pos_encodings)
		dense1 = self.dense1(dense0)
		dense2 = self.dense2(dense1)
		dense3 = self.dense3(dense2)

		dense4 = self.dense4(torch.cat([pos_encodings, dense3], dim=-1))
		dense5 = self.dense5(dense4)
		dense6 = self.dense6(dense5)
		dense7 = self.dense7(dense6)

		if self.appearance:
			dense8 = self.dense8(torch.cat([dense7, dir_encodings, app_encodings], dim=-1))
		else:
			dense8 = self.dense8(torch.cat([dense7, dir_encodings], dim=-1))

		dense9 = self.dense9(dense8)

		results = torch.add(torch.mul(dense9, 0.2), colors_c2)

		return results


class Res_Block(nn.Module):
	def __init__(self, units_cnn):
		super(Res_Block, self).__init__()
		self.units_cnn = units_cnn
		self.build()

	def build(self):
		self.conv0 = nn.Sequential(nn.Conv2d(self.units_cnn, self.units_cnn, kernel_size=3, stride=1, 
			padding=1, padding_mode="reflect"), nn.ReLU()).to(device)
		self.conv1 = nn.Conv2d(self.units_cnn, self.units_cnn, kernel_size=3, stride=1, 
			padding=1, padding_mode="reflect", device=device)

	def forward(self, inputs):
		conv0 = self.conv0(inputs)
		conv1 = self.conv1(conv0)
		results = torch.add(inputs, conv1)
		return results


class Render_Refinement(nn.Module):
	def __init__(self, units_cnn):
		super(Render_Refinement, self).__init__()
		self.units_cnn = units_cnn
		self.build()

	def build(self):
		self.res_block0 = Res_Block(self.units_cnn)
		self.res_block1 = Res_Block(self.units_cnn)

		self.features = nn.Sequential(nn.Conv2d(3, self.units_cnn, kernel_size=3, stride=1, 
			padding=1, padding_mode="reflect"), nn.ReLU()).to(device)
		self.conv1 = nn.Conv2d(self.units_cnn, self.units_cnn, kernel_size=3, stride=1, 
			padding=1, padding_mode="reflect", device=device)
		self.conv2 = nn.Conv2d(self.units_cnn, self.units_cnn, kernel_size=3, stride=1, 
			padding=1, padding_mode="reflect", device=device)
		self.conv3 = nn.Sequential(nn.Conv2d(self.units_cnn, self.units_cnn, kernel_size=3, stride=1, 
			padding=1, padding_mode="reflect"), nn.ReLU()).to(device)
		self.conv4 = nn.Sequential(nn.Conv2d(self.units_cnn, 3, kernel_size=3, stride=1, 
			padding=1, padding_mode="reflect"), nn.Sigmoid()).to(device)

	def forward(self, inputs):
		inputs = torch.movedim(inputs, -1, 1)
		conv0 = self.features(inputs)
		conv1 = self.conv1(conv0)
		res0 = self.res_block0(conv1)
		res1 = self.res_block1(res0)
		conv2 = self.conv2(res1)
		conv3 = self.conv3(torch.add(conv1, conv2))
		conv4 = self.conv4(conv3)
		results = torch.movedim(conv4, 1, -1)
		return results


class App_Embeddings(nn.Module):
	def __init__(self, num_poses, app_dims):
		super(App_Embeddings, self).__init__()
		self.embeddings = nn.Embedding(num_poses, embedding_dim=app_dims, device=device)

	def forward(self, indic):
		results = self.embeddings(indic)
		return results