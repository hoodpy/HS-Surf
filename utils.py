import torch
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")

class Position_Encoder(nn.Module):
	def __init__(self, levels):
		super(Position_Encoder, self).__init__()
		self.levels = levels

	def forward(self, means, covs):
		means = torch.reshape(means, [-1, 3])
		covs = torch.reshape(covs, [-1, 3])
		encodings = [means]

		for i in range(self.levels):
			med0 = torch.mul(torch.pow(torch.tensor(2.0), i).to(dtype=torch.float32), means)
			med1 = torch.exp(torch.mul(torch.mul(-0.5, torch.pow(torch.tensor(4.0), i)), covs))
			sin_ = torch.mul(torch.sin(med0), med1)
			cos_ = torch.mul(torch.cos(med0), med1)
			encodings.append(torch.cat([sin_, cos_], dim=-1))

		encodings = torch.cat(encodings, dim=-1)

		return encodings


class Point_Encoder(nn.Module):
	def __init__(self, levels):
		super(Point_Encoder, self).__init__()
		self.levels = levels

	def forward(self, poi_xyz):
		poi_xyz = torch.reshape(poi_xyz, [-1, 3])
		encodings = [poi_xyz]

		for i in range(self.levels):
			med = torch.mul(torch.mul(torch.pow(torch.tensor(2.0), i), np.pi), poi_xyz)
			sin_, cos_ = torch.sin(med), torch.cos(med)
			encodings.append(torch.cat([sin_, cos_], dim=-1))

		encodings = torch.cat(encodings, dim=-1)

		return encodings


class Direction_Encoder(nn.Module):
	def __init__(self, levels):
		super(Direction_Encoder, self).__init__()
		self.levels = levels

	def forward(self, dir_xyz):
		dir_xyz = torch.reshape(dir_xyz, [-1, 3])
		encodings = [dir_xyz]

		for i in range(self.levels):
			med = torch.mul(torch.mul(torch.pow(torch.tensor(2.0), i), np.pi), dir_xyz)
			sin_, cos_ = torch.sin(med), torch.cos(med)
			encodings.append(torch.cat([sin_, cos_], dim=-1))

		encodings = torch.cat(encodings, dim=-1)

		return encodings


class Volume_Render(nn.Module):
	def __init__(self):
		super(Volume_Render, self).__init__()

	def compute_weights(self, sigmas, t_inters):
		eps, inf = 1e-10, 1e10
		t_centers = torch.mean(t_inters, dim=-1)

		diffs = torch.sub(t_centers[:, 1:], t_centers[:, :-1])
		last_array = torch.ones_like(diffs[:, :1], dtype=torch.float32) * inf
		diffs = torch.cat([diffs, last_array], dim=-1)

		alphas = 1.0 - torch.exp(-1.0 * torch.mul(sigmas, diffs))
		temp = torch.cumprod(1.0 - alphas[:, :-1] + eps, dim=-1)
		alpha_weights = torch.cat([torch.ones_like(temp[:, :1], dtype=torch.float32), temp], dim=-1)

		weights = torch.mul(alphas, alpha_weights)

		return weights, t_centers

	def forward(self, sigmas, rgbs, t_inters):
		weights, t_centers = self.compute_weights(sigmas, t_inters)

		depths = torch.sum(torch.mul(weights, t_centers), dim=1)
		colors = torch.sum(torch.mul(torch.unsqueeze(weights, dim=-1), rgbs), dim=1)

		return depths, colors, weights


class Sample_PDF(nn.Module):
	def __init__(self, num_rays, inters_fine):
		super(Sample_PDF, self).__init__()
		self.num_rays = num_rays
		self.inters_fine = inters_fine

	def forward(self, weights, t_inters):
		eps = 1e-5

		weights_pad = torch.cat([weights[:, :1], weights, weights[:, -1:]], dim=-1)
		weights_max = torch.maximum(weights_pad[:, :-1], weights_pad[:, 1:])
		weights_blur = torch.div(torch.add(weights_max[:, :-1], weights_max[:, 1:]), 2.0)
		weights = torch.add(weights_blur, 0.01)

		t_vals = torch.cat([t_inters[..., 0], t_inters[:, -1:, 1]], dim=-1)

		weights_sum = torch.sum(weights, dim=-1, keepdim=True)
		padding = torch.maximum(torch.tensor(0.0), eps - weights_sum)

		weights = weights + padding / weights.shape[-1]
		weights_sum = weights_sum + padding

		pdf = weights / weights_sum #[N, 64]
		cdf = torch.minimum(torch.tensor(1.0), torch.cumsum(pdf[..., :-1], dim=-1))

		first_array = torch.zeros([self.num_rays, 1], dtype=torch.float32, device=device)
		last_array = torch.ones([self.num_rays, 1], dtype=torch.float32, device=device)
		cdf = torch.cat([first_array, cdf, last_array], dim=-1) #[N, 65]

		s = 1.0 / (self.inters_fine+1)
		u = np.arange(self.inters_fine+1) * s
		u = np.broadcast_to(u, [self.num_rays, self.inters_fine+1])

		jitter = np.random.uniform(high=s-np.finfo("float32").eps, size=[self.num_rays, self.inters_fine+1])
		u = u + jitter
		u = np.minimum(u, 1.0 - np.finfo("float32").eps) #[N, 32]
		u = torch.from_numpy(u).to(cdf)

		mask = u[..., None, :] >= cdf[..., None] #[N, 65, 32]

		bins_g0 = torch.max(torch.where(mask, t_vals[..., None], t_vals[:, :1, None]), dim=1)[0] #[N, 32]
		bins_g1 = torch.min(torch.where(~mask, t_vals[..., None], t_vals[:, -1:, None]), dim=1)[0]

		cdf_g0 = torch.max(torch.where(mask, cdf[..., None], cdf[:, :1, None]), dim=1)[0]
		cdf_g1 = torch.min(torch.where(~mask, cdf[..., None], cdf[:, -1:, None]), dim=1)[0]

		t = (u - cdf_g0) / (cdf_g1 - cdf_g0)
		t[t != t] = 0
		t = torch.clamp(t, 0.0, 1.0)

		t_vals = bins_g0 + t * (bins_g1 - bins_g0)
		t_vals = t_vals.detach()
		t_vals, _ = torch.sort(t_vals, dim=-1, descending=False)

		t_inters = torch.stack([t_vals[:, :-1], t_vals[:, 1:]], dim=-1)

		return t_inters


class Distor_Value(nn.Module):
	def __init__(self):
		super(Distor_Value, self).__init__()

	def forward(self, t_inters, weights, t_near, t_far):
		t_vals = torch.cat([t_inters[..., 0], t_inters[:, -1:, 1]], dim=-1)
		s_vals = torch.div(torch.sub(t_vals, t_near), torch.sub(t_far, t_near))

		ut = torch.div(torch.add(s_vals[:, 1:], s_vals[:, :-1]), 2.0)
		dut = torch.abs(torch.sub(torch.unsqueeze(ut, dim=-1), torch.unsqueeze(ut, dim=1))) #[N, 32, 32]

		loss_inter = torch.sum(torch.mul(weights, torch.sum(torch.mul(torch.unsqueeze(weights, 
			dim=1), dut), dim=-1)), dim=-1)

		loss_intra = torch.sum(torch.mul(torch.square(weights), torch.sub(s_vals[:, 1:], s_vals[:, :-1])), dim=-1)

		value = torch.add(loss_inter, torch.div(loss_intra, 3.0))

		return value


class Get_Block_Mask(nn.Module):
	def __init__(self, box_min, box_max, grid_x, grid_y):
		super(Get_Block_Mask, self).__init__()
		self.blocks = grid_x * grid_y
		self.box_min = torch.tensor(box_min, dtype=torch.float32, device=device)
		self.box_max = torch.tensor(box_max, dtype=torch.float32, device=device)
		self.grid_x = grid_x
		self.grid_y = grid_y
		self.build()

	def build(self):
		bounding_box = torch.sub(self.box_max[0, :2], self.box_min[0, :2])
		size_x = torch.div(bounding_box[0], self.grid_x)
		size_y = torch.div(bounding_box[1], self.grid_y)
		self.box_mins = []
		self.box_maxs = []

		for i in range(self.grid_x):
			for j in range(self.grid_y):
				start_x = torch.add(self.box_min[0, 0], size_x * i)
				start_y = torch.add(self.box_min[0, 1], size_y * j)
				stop_x, stop_y = torch.add(start_x, size_x), torch.add(start_y, size_y)
				self.box_mins.append(torch.tensor([[start_x, start_y]], dtype=torch.float32, device=device))
				self.box_maxs.append(torch.tensor([[stop_x, stop_y]], dtype=torch.float32, device=device))

	def forward(self, pos_xyz):
		pos_xyz, results = pos_xyz[:, :2], []

		for i in range(self.blocks):
			box_min, box_max = self.box_mins[i], self.box_maxs[i]
			pos_clip = torch.maximum(torch.minimum(pos_xyz, box_max), box_min)
			mask = torch.eq(pos_clip, pos_xyz).to(dtype=torch.int32)
			mask = torch.eq(torch.sum(mask, dim=-1, keepdim=True), 2).to(dtype=torch.float32)
			results.append(mask)

		results = torch.cat(results, dim=-1) #[N_rays, 8]

		return results


class Merge_Depths(nn.Module):
	def __init__(self):
		super(Merge_Depths, self).__init__()

	def forward(self, depths_fs, block_mask):
		depths_fs = torch.sum(torch.mul(depths_fs, block_mask), dim=-1, keepdim=True)
		results = torch.div(depths_fs, torch.sum(block_mask, dim=-1, keepdim=True))
		return results


class Get_Inputs_Fine(nn.Module):
	def __init__(self, box_center):
		super(Get_Inputs_Fine, self).__init__()
		self.box_center = torch.tensor(box_center, dtype=torch.float32, device=device)

	def forward(self, rays_o, rays_d, radii, depths, t_range, t_near, t_far):
		rays_o = torch.sub(rays_o, self.box_center)
		radii = torch.squeeze(radii, dim=-1)

		left = torch.maximum(torch.sub(depths, torch.mul(t_range, 0.5)), t_near)
		right = torch.minimum(torch.add(depths, torch.mul(t_range, 0.5)), t_far)
		inter = torch.cat([left, right], dim=-1)

		c = torch.mean(inter, dim=-1)
		d = torch.div(torch.sub(inter[..., 1], inter[..., 0]), 2.0)

		t_mean = torch.add(c, torch.div(2.0 * torch.mul(c, torch.square(d)), torch.add(3.0 * torch.square(c), 
			torch.square(d))))

		t_var0 = torch.div(torch.square(d), 3.0)
		t_var1 = torch.div(torch.mul(torch.pow(d, 4), torch.sub(12. * torch.square(c), torch.square(d))), 
			torch.square(torch.add(3.0 * torch.square(c), torch.square(d))))
		t_var = torch.sub(t_var0, 4.0 / 15.0 * t_var1)

		r_var = 4.0 / 15.0 * torch.div(torch.pow(d, 4), torch.add(3.*torch.square(c), torch.square(d)))
		r_var = torch.mul(torch.square(radii), 1.0 / 4.0 * torch.square(c) + 5.0 / 12.0 * torch.square(d) - r_var)

		means = torch.add(rays_o, torch.mul(torch.unsqueeze(t_mean, dim=-1), rays_d))

		null_outer_diag = 1.0 - torch.square(rays_d) / torch.sum(torch.square(rays_d), dim=-1, keepdim=True)

		covs = torch.add(torch.mul(torch.unsqueeze(t_var, dim=-1), torch.square(rays_d)), torch.mul(
			torch.unsqueeze(r_var, dim=-1), null_outer_diag))

		return means, covs


class Merge_Colors(nn.Module):
	def __init__(self):
		super(Merge_Colors, self).__init__()

	def forward(self, colors_fs, block_mask):
		colors_fs = torch.mul(colors_fs, torch.unsqueeze(block_mask, dim=-1))
		results = torch.div(torch.sum(colors_fs, dim=1), torch.sum(block_mask, dim=-1, keepdim=True))
		return results


class Get_Colors(nn.Module):
	def __init__(self, width):
		super(Get_Colors, self).__init__()
		self.width = width

	def forward(self, coords, image):
		colors_r, colors_g, colors_b = image[..., 0], image[..., 1], image[..., 2]
		indic = torch.add(torch.mul(coords[:, 0], self.width), coords[:, 1]).long()
		colors_r = torch.take(colors_r, indic)
		colors_g = torch.take(colors_g, indic)
		colors_b = torch.take(colors_b, indic)
		colors_rgb = torch.stack([colors_r, colors_g, colors_b], dim=-1)
		return colors_rgb


class Depth_Mask(nn.Module):
	def __init__(self):
		super(Depth_Mask, self).__init__()

	def forward(self, coords):
		mask = torch.add(coords, 2) % 3 #[N_rays, 2]
		mask = torch.sum(mask, dim=-1) #[N_rays]
		mask = torch.eq(mask, 0).to(dtype=torch.float32)
		return mask