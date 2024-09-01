import torch
import torch.nn as nn
import numpy as np
import ray_utils
import mlp_utils
import utils

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")

class NetWork(nn.Module):
	def __init__(self, trainable, mode, params):
		super(NetWork, self).__init__()
		self.trainable = trainable
		self.mode = mode
		self.num_rays = params["num_rays"]
		self.inters_proposal = params["inters_proposal"]
		self.inters_property = params["inters_property"]
		self.pos_levels = params["pos_levels"]
		self.poi_levels = params["poi_levels"]
		self.dir_levels = params["dir_levels"]
		self.units_nerf = params["units_nerf"]
		self.units_dec = params["units_dec"]
		self.units_ger = params["units_ger"]
		self.units_mlp = params["units_mlp"]
		self.units_cnn = params["units_cnn"]
		self.box_min = params["box_min"]
		self.box_max = params["box_max"]
		self.grid_x = params["grid_x"]
		self.grid_y = params["grid_y"]
		self.hash_base = params["hash_base"]
		self.hash_finest = params["hash_finest"]
		self.hash_scale = params["hash_scale"]
		self.hash_levels = params["hash_levels"]
		self.units_ngp = params["units_ngp"]
		self.hash_size = params["hash_size"]
		self.app_dims = params["app_dims"]
		self.appearance = params["appearance"]
		self.nerf = params["nerf"]
		self.cnn = params["cnn"]
		self.build(params)

	def build(self, params):
		self.blocks = self.grid_x * self.grid_y
		self.get_rays = ray_utils.Get_Rays(params["high"], params["width"], self.num_rays)

		if self.mode == "earth":
			self.get_inters = ray_utils.Get_Inters_Earth(self.inters_proposal, params["scene_scale"])
		elif self.mode == "planes":
			self.get_inters = ray_utils.Get_Inters_Plane(self.inters_proposal, params["z_near"], params["z_far"], params["max_len"])
		elif self.mode == "blender":
			self.get_inters = ray_utils.Get_Inters_Blender(self.num_rays, self.inters_proposal, params["p_near"], params["p_far"])
		else:
			raise ValueError("Note: mode must be in [earth, planes, blender]")

		self.get_inputs = ray_utils.Get_Inputs()

		self.pos_encoder = utils.Position_Encoder(self.pos_levels)
		self.poi_encoder = utils.Point_Encoder(self.poi_levels)
		self.dir_encoder = utils.Direction_Encoder(self.dir_levels)

		self.hash_encoder = mlp_utils.Hash_Encoder(self.box_min, self.box_max, self.hash_base, self.hash_finest, 
			self.hash_scale, self.units_nerf, self.units_ngp, self.hash_levels, self.hash_size)

		self.get_inters_property = mlp_utils.Get_Inters_Property(self.num_rays, self.pos_levels, self.dir_levels, 
			self.units_nerf, self.app_dims, self.appearance)

		self.volume_render = utils.Volume_Render()
		self.sample_pdf = utils.Sample_PDF(self.num_rays, self.inters_property)
		self.distor_value = utils.Distor_Value()

		self.get_block_mask = utils.Get_Block_Mask(self.box_min, self.box_max, self.grid_x, self.grid_y)

		box_mins, box_maxs = [], []
		size_x = (self.box_max - self.box_min)[0, 0] / self.grid_x
		size_y = (self.box_max - self.box_min)[0, 1] / self.grid_y

		for i in range(self.grid_x):
			for j in range(self.grid_y):
				start_x = self.box_min[0, 0] + i * size_x
				start_y = self.box_min[0, 1] + j * size_y
				stop_x, stop_y = start_x + size_x, start_y + size_y
				box_mins.append(np.array([[start_x, start_y, 0.0]], dtype=np.float32))
				box_maxs.append(np.array([[stop_x, stop_y, 0.0]], dtype=np.float32))

		box_centers = [(box_mins[i] + box_maxs[i]) / 2.0 for i in range(self.blocks)]

		self.depths_correct = nn.ModuleList([mlp_utils.Depth_Correct(box_center, self.poi_levels, self.dir_levels, 
			self.units_dec) for box_center in box_centers])

		self.merge_depths = utils.Merge_Depths()
		self.get_range = mlp_utils.Get_Range(self.units_ger)

		self.get_inputs_fines = nn.ModuleList([utils.Get_Inputs_Fine(box_center) for box_center in box_centers])

		self.get_fines_property = nn.ModuleList([mlp_utils.Get_Fine_Property(self.num_rays, self.pos_levels, self.dir_levels, 
			self.units_mlp, self.app_dims, self.appearance) for _ in range(self.blocks)])

		self.merge_colors = utils.Merge_Colors()
		self.render_refinement = mlp_utils.Render_Refinement(self.units_cnn)

		self.hash_encoder.requires_grad_(self.nerf)
		self.get_inters_property.requires_grad_(self.nerf)
		self.depths_correct.requires_grad_(self.nerf)
		self.get_range.requires_grad_(self.nerf)
		self.get_fines_property.requires_grad_(self.nerf)
		self.render_refinement.requires_grad_(self.cnn)

	def forward_stage0(self, coords, pose, focal, app_encodings):
		rays_o, rays_d, radii = self.get_rays(coords, pose, focal)
		t_inters_proposal, t_near, t_far = self.get_inters(rays_o, rays_d)

		means, covs, poi_xyz, dir_xyz = self.get_inputs(rays_o, rays_d, radii, t_inters_proposal, self.inters_proposal)

		pos_encodings = self.pos_encoder(means, covs)
		dir_encodings = self.dir_encoder(dir_xyz)
		gemo_encodings, color_encodings, mask = self.hash_encoder(poi_xyz)

		sigmas, rgbs = self.get_inters_property(pos_encodings, dir_encodings, app_encodings, gemo_encodings, 
			color_encodings, mask, self.inters_proposal)

		depths_c1, colors_c1, weights_proposal = self.volume_render(sigmas, rgbs, t_inters_proposal)

		t_inters = self.sample_pdf(weights_proposal, t_inters_proposal)

		means, covs, poi_xyz, dir_xyz = self.get_inputs(rays_o, rays_d, radii, t_inters, self.inters_property)

		pos_encodings = self.pos_encoder(means, covs)
		dir_encodings = self.dir_encoder(dir_xyz)
		gemo_encodings, color_encodings, mask = self.hash_encoder(poi_xyz)

		sigmas, rgbs = self.get_inters_property(pos_encodings, dir_encodings, app_encodings, gemo_encodings, 
			color_encodings, mask, self.inters_property)

		depths_c2, colors_c2, weights = self.volume_render(sigmas, rgbs, t_inters)

		distor_value = self.distor_value(t_inters, weights, t_near, t_far) if self.trainable else None

		poi_xyz = torch.add(rays_o, torch.mul(torch.unsqueeze(depths_c2, dim=-1), rays_d))

		dir_encodings = self.dir_encoder(rays_d)
		block_mask = self.get_block_mask(poi_xyz) #[N_rays, 8]

		depths_fs = [self.depths_correct[i](poi_xyz, dir_encodings, depths_c2) for i in range(self.blocks)]
		depths_fs = torch.clamp(torch.cat(depths_fs, dim=-1), min=t_near, max=t_far) #[N_rays, 8]

		depths_f1 = self.merge_depths(depths_fs, block_mask) #[N_rays, 1]
		s_vals = torch.div(torch.sub(depths_f1, t_near), torch.sub(t_far, t_near))
		t_range = self.get_range(t_near, t_far, s_vals)

		poi_xyz = torch.add(rays_o, torch.mul(depths_f1, rays_d))
		block_mask = self.get_block_mask(poi_xyz)

		colors_fs = []

		for i in range(self.blocks):
			means, covs = self.get_inputs_fines[i](rays_o, rays_d, radii, depths_f1, t_range, t_near, t_far)
			colors_fs.append(self.get_fines_property[i](means, covs, dir_encodings, app_encodings, colors_c2))

		colors_fs = [torch.unsqueeze(colors_f, dim=1) for colors_f in colors_fs]
		colors_f1 = self.merge_colors(torch.cat(colors_fs, dim=1), block_mask)
		depths_f1 = torch.squeeze(depths_f1, dim=-1)

		return depths_c1, depths_c2, colors_c1, colors_c2, depths_f1, colors_f1, distor_value

	def forward_stage1(self, colors_f1):
		colors_f2 = self.render_refinement(colors_f1)
		return colors_f2