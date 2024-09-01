import torch
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")

class Get_Rays(nn.Module):
	def __init__(self, high, width, num_rays):
		super(Get_Rays, self).__init__()
		self.high = high
		self.width = width
		self.num_rays = num_rays
		self.build()

	def build(self):
		w_list, h_list = torch.meshgrid(torch.arange(self.width), torch.arange(self.high), indexing="xy")
		self.hw_list = torch.stack([h_list, w_list], dim=-1).to(dtype=torch.float32).to(device)

	def forward(self, coords, pose, focal):
		xs = torch.div(torch.sub(self.hw_list[..., 1], 0.5 * self.width), focal)
		ys = torch.div(torch.sub(self.hw_list[..., 0], 0.5 * self.high), -1.0 * focal)
		zs = torch.mul(torch.ones_like(xs, dtype=torch.float32, device=device), -1.0)

		directions = torch.stack([xs, ys, zs], dim=-1)
		directions = torch.div(directions, torch.norm(directions, dim=-1, keepdim=True))

		rays_o = torch.tile(torch.unsqueeze(pose[:, 3], dim=0), [self.num_rays, 1])

		rays_d = torch.sum(torch.mul(torch.unsqueeze(directions, dim=2), torch.unsqueeze(torch.unsqueeze(
			pose[:, :3], dim=0), dim=0)), dim=-1)
		rays_d = torch.div(rays_d, torch.norm(rays_d, dim=-1, keepdim=True))

		dh = torch.norm(torch.sub(rays_d[:-1], rays_d[1:]), dim=-1, keepdim=False)
		dh = torch.cat([dh, dh[-2:-1]], dim=0) #[H, W]

		radii = torch.mul(dh, torch.div(2., torch.sqrt(torch.tensor(12.))).to(dtype=torch.float32))

		indic = torch.add(coords[:, 1], torch.mul(coords[:, 0], self.width)).long()
		radii = torch.unsqueeze(torch.take(radii, indic), dim=-1)

		coords = torch.add(coords.to(dtype=torch.float32), 0.5)

		xs = torch.div(torch.sub(coords[:, 1], 0.5 * self.width), focal)
		ys = torch.div(torch.sub(coords[:, 0], 0.5 * self.high), -1.0 * focal)
		zs = torch.mul(torch.ones_like(xs, dtype=torch.float32), -1.0)

		directions = torch.stack([xs, ys, zs], dim=-1)
		directions = torch.div(directions, torch.norm(directions, dim=-1, keepdim=True))

		rays_d = torch.sum(torch.mul(torch.unsqueeze(directions, dim=1), torch.unsqueeze(pose[:, :3], dim=0)), dim=-1)
		rays_d = torch.div(rays_d, torch.norm(rays_d, dim=-1, keepdim=True))

		return rays_o, rays_d, radii


class Get_Inters_Earth(nn.Module):
	def __init__(self, inters, scene_scale):
		super(Get_Inters_Earth, self).__init__()
		self.inters = inters
		self.scene_scale = scene_scale
		self.build()

	def build(self):
		self.global_center = torch.tensor(np.array([[0.,0.,-6371011.]])*self.scene_scale, dtype=torch.float32, 
			requires_grad=False, device=device)
		self.earth_rad = torch.tensor(6371011*self.scene_scale,dtype=torch.float32,requires_grad=False).to(device)
		self.earth_full = torch.tensor((6371011+250)*self.scene_scale,dtype=torch.float32,requires_grad=False).to(device)

	def get_near_far(self, rays_o, rays_d):
		a = torch.sum(torch.square(rays_d), dim=-1) #[N]
		b = 2. * torch.sum(torch.mul(torch.sub(rays_o, self.global_center), rays_d), dim=-1) #[N]

		c_near = torch.sub(torch.sum(torch.square(torch.sub(rays_o, self.global_center)), dim=-1), 
			torch.square(self.earth_full))
		c_far = torch.sub(torch.sum(torch.square(torch.sub(rays_o, self.global_center)), dim=-1), 
			torch.square(self.earth_rad))

		delta_near = torch.sqrt(torch.sub(torch.square(b), 4. * torch.mul(a, c_near)))
		delta_far = torch.sqrt(torch.sub(torch.square(b), 4. * torch.mul(a, c_far)))

		d_near = torch.div(torch.sub(-1. * b, delta_near), 2. * a)
		d_far = torch.div(torch.sub(-1. * b, delta_far), 2. * a)

		near = torch.unsqueeze(torch.mul(d_near, 0.9), dim=-1)
		far = torch.unsqueeze(torch.mul(d_far, 1.1), dim=-1)

		return near, far

	def forward(self, rays_o, rays_d):
		t_near, t_far = self.get_near_far(rays_o, rays_d)

		s_vals = torch.unsqueeze(torch.linspace(start=0., end=1., steps=self.inters+1, dtype=torch.float32), dim=0).to(device)
		t_vals = 1.0 / torch.add(torch.mul(1.0 / t_near, 1.0 - s_vals), torch.mul(1.0 / t_far, s_vals))
		t_vals_half = t_vals[..., :int(self.inters*2.0/3.0)]

		linear_start = t_vals_half[..., -1:]
		linear_vals = torch.linspace(start=0., end=1., steps=self.inters-int(self.inters*2.0/3.0)+2, dtype=torch.float32)
		linear_vals = torch.unsqueeze(linear_vals, dim=0).to(device)

		linear_half = torch.add(torch.mul(linear_start, 1.0 - linear_vals), torch.mul(t_far, linear_vals))
		t_vals_tol, _ = torch.sort(torch.cat([t_vals_half, linear_half[..., 1:]], dim=-1), dim=-1, descending=False)

		t_inters = torch.stack([t_vals_tol[..., :-1], t_vals_tol[..., 1:]], dim=-1)

		return t_inters, t_near, t_far


class Get_Inters_Plane(nn.Module):
	def __init__(self, inters, z_near, z_far, max_len):
		super(Get_Inters_Plane, self).__init__()
		self.inters = inters
		self.z_near = z_near
		self.z_far = z_far
		self.max_len = max_len

	def get_near_far(self, rays_o, rays_d):
		high2near = torch.maximum(torch.sub(rays_o[:, 2], self.z_near), torch.zeros_like(rays_o[:, 2]))
		high2far = torch.maximum(torch.sub(rays_o[:, 2], self.z_far), torch.zeros_like(rays_o[:, 2]))

		t_near = torch.div(high2near, torch.abs(rays_d[:, 2]) + 1e-6)
		t_far = torch.div(high2far, torch.abs(rays_d[:, 2]) + 1e-6)
		t_far = torch.minimum(t_far, torch.add(t_near, self.max_len))

		t_near = torch.unsqueeze(t_near, dim=-1)
		t_far = torch.unsqueeze(t_far, dim=-1)

		return t_near, t_far

	def forward(self, rays_o, rays_d):
		t_near, t_far = self.get_near_far(rays_o, rays_d)

		s_vals = torch.linspace(start=0.0, end=1.0, steps=self.inters+1, dtype=torch.float32, device=device)
		s_vals = torch.unsqueeze(s_vals, dim=0)

		t_vals = torch.add(torch.mul(t_near, 1.0 - s_vals), torch.mul(t_far, s_vals))
		t_vals_tol, _ = torch.sort(t_vals, dim=-1, descending=False)

		t_inters = torch.stack([t_vals_tol[..., :-1], t_vals_tol[..., 1:]], dim=-1)

		return t_inters, t_near, t_far


class Get_Inters_Blender(nn.Module):
	def __init__(self, num_rays, inters, p_near, p_far):
		super(Get_Inters_Blender, self).__init__()
		self.num_rays = num_rays
		self.inters = inters
		self.p_near = p_near
		self.p_far = p_far

	def forward(self, rays_o, rays_d):
		s_vals = torch.linspace(start=0.0, end=1.0, steps=self.inters+1, dtype=torch.float32, device=device)
		t_vals = torch.unsqueeze(torch.add(torch.mul(1.0 - s_vals, self.p_near), torch.mul(s_vals, self.p_far)), dim=0)

		t_mids = torch.div(torch.add(t_vals[:, :-1], t_vals[:, 1:]), 2.0)

		lower = torch.cat([t_vals[:, :1], t_mids], dim=-1)
		upper = torch.cat([t_mids, t_vals[:, -1:]], dim=-1)

		s_rand = torch.rand([self.num_rays, self.inters+1], dtype=torch.float32, device=device)

		t_vals = torch.add(lower, torch.mul(torch.sub(upper, lower), s_rand))
		t_inters = torch.stack([t_vals[:, :-1], t_vals[:, 1:]], dim=-1)

		t_near = torch.full([self.num_rays, 1], fill_value=self.p_near, dtype=torch.float32, device=device)
		t_far = torch.full([self.num_rays, 1], fill_value=self.p_far, dtype=torch.float32, device=device)

		return t_inters, t_near, t_far


class Get_Inputs(nn.Module):
	def __init__(self):
		super(Get_Inputs, self).__init__()

	def forward(self, rays_o, rays_d, radii, t_inters, inters):
		c = torch.mean(t_inters, dim=-1)
		d = torch.div(torch.sub(t_inters[..., 1], t_inters[..., 0]), 2.0)

		t_mean = torch.add(c, torch.div(2.0 * torch.mul(c, torch.square(d)), torch.add(3.0 * torch.square(c), 
			torch.square(d))))

		t_var0 = torch.div(torch.square(d), 3.0)
		t_var1 = torch.div(torch.mul(torch.pow(d, 4), torch.sub(12. * torch.square(c), torch.square(d))), 
			torch.square(torch.add(3.0 * torch.square(c), torch.square(d))))
		t_var = torch.sub(t_var0, 4.0 / 15.0 * t_var1)

		r_var = 4.0 / 15.0 * torch.div(torch.pow(d, 4), torch.add(3.*torch.square(c), torch.square(d)))
		r_var = torch.mul(torch.square(radii), 1.0 / 4.0 * torch.square(c) + 5.0 / 12.0 * torch.square(d) - r_var)

		dir_xyz = torch.tile(torch.unsqueeze(rays_d, dim=1), [1, inters, 1])
		means = torch.add(torch.unsqueeze(rays_o, dim=1), torch.mul(torch.unsqueeze(t_mean, dim=-1), dir_xyz))

		null_outer_diag = 1.0 - torch.square(dir_xyz) / torch.sum(torch.square(dir_xyz), dim=-1, keepdim=True)
		covs = torch.add(torch.mul(torch.unsqueeze(t_var, dim=-1), torch.square(dir_xyz)), torch.mul(
			torch.unsqueeze(r_var, dim=-1), null_outer_diag))

		pos_xyz = torch.add(torch.unsqueeze(rays_o, dim=1), torch.mul(torch.unsqueeze(c, dim=-1), dir_xyz))

		return means, covs, pos_xyz, dir_xyz