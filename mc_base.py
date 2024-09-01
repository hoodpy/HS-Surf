import torch
import torch.nn as nn
import numpy as np
import json
import time
import cv2
import os
import utils
import mlp_utils
import datasets
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from network import NetWork

torch.set_default_dtype(torch.float32)
device = torch.device("cuda")

def post_display(image):
	vmax = np.percentile(image, 95)
	normalizer = mpl.colors.Normalize(vmin=image.min(), vmax=vmax)
	mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
	colormapped_im = (mapper.to_rgba(image)[:, :, :3] * 255).astype(np.uint8)
	return colormapped_im


class MCBase_NeRF():
	def __init__(self, configs, trainable=True, rende_only=False, stage=0, init_epoch=0):
		self.trainable = trainable
		self.stage = stage
		self.init_epoch = init_epoch
		self.num_rays = configs.num_rays_test if rende_only else configs.num_rays
		self.inters_proposal = configs.inters_proposal
		self.inters_property = configs.inters_property
		self.batch_size = configs.batch_size
		self.hash_base = configs.hash_base
		self.hash_finest = configs.hash_finest
		self.hash_scale = configs.hash_scale
		self.hash_levels = configs.hash_levels
		self.hash_size = configs.hash_size
		self.grid_x = configs.grid_x
		self.grid_y = configs.grid_y
		self.max_rays = configs.max_rays
		self.max_images = configs.max_images
		self.max_rays_test = configs.max_rays_test
		self.lr_nerf = configs.lr_nerf
		self.lr_cnn = configs.lr_cnn
		self.saves_min = configs.saves_min
		self.appearance = configs.appearance
		self.app_dims = configs.app_dims
		self.units_ngp = configs.units_ngp
		self.units_nerf = configs.units_nerf
		self.units_dec = configs.units_dec
		self.units_ger = configs.units_ger
		self.units_mlp = configs.units_mlp
		self.units_cnn = configs.units_cnn
		self.pos_levels = configs.pos_levels
		self.poi_levels = configs.poi_levels
		self.dir_levels = configs.dir_levels
		self.stride_stage1 = configs.stride_stage1
		self.mode = configs.mode
		self.data = configs.data
		self.load_file_path()
		self.data_dict = self.load_data_dict()
		self.prepare_for_render()
		self.network = NetWork(self.trainable, self.mode, self.data_dict)
		self.get_colors = utils.Get_Colors(self.data_dict["width"])
		self.depth_mask = utils.Depth_Mask()
		self.loss_l1 = nn.L1Loss().to(device)
		self.loss_l2 = nn.MSELoss().to(device)
		self.network_variables_initializer()
		self.load_weights()

	def load_file_path(self):
		if self.trainable:
			if self.data in ["data", "google", "building", "rubble", "residence", "campus"]:
				self.image_path = "./data/%s/images/" % (self.data)
				self.json_path = "./data/%s/train.json" % (self.data)

			elif self.data in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]:
				self.image_path = "./data/nerf_synthetic/%s/train/" % (self.data)
				self.json_path = "./data/nerf_synthetic/%s/transforms_train.json" % (self.data)

			else:
				raise ValueError("Can not find the corresponding files.")

		else:
			if self.data in ["data", "google", "building", "rubble", "residence", "campus"]:
				self.image_path = "./data/%s/images/" % (self.data)
				self.json_path = "./data/%s/test.json" % (self.data)

			elif self.data in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]:
				self.image_path = "./data/nerf_synthetic/%s/test/" % (self.data)
				self.json_path = "./data/nerf_synthetic/%s/transforms_test.json" % (self.data)

			else:
				raise ValueError("Can not find the corresponding files.")

		self.base_path = "./experiments/" + self.data
		self.model_path = self.base_path + "/model"
		self.results_path = "./reconstructions/" + self.data

		self.box_path = self.base_path + "/box.json"
		self.nerf_model = self.model_path + "/nerf.pth"
		self.cnn_model = self.model_path + "/cnn.pth"
		self.app_train = self.model_path + "/app_train.pth"
		self.app_test = self.model_path + "/app_test.pth"
		self.log_path = self.base_path + "/log"
		self.config_path = self.base_path + "/configs.txt"
		self.colors_m = self.base_path + "/colors_m/"

		self.depths_c1 = self.results_path + "/depths_c1/"
		self.depths_c2 = self.results_path + "/depths_c2/"
		self.colors_c1 = self.results_path + "/colors_c1/"
		self.colors_c2 = self.results_path + "/colors_c2/"
		self.depths_f1 = self.results_path + "/depths_f1/"
		self.colors_f1 = self.results_path + "/colors_f1/"
		self.colors_f2 = self.results_path + "/colors_f2/"

		if not os.path.exists(self.base_path): os.makedirs(self.base_path)
		if not os.path.exists(self.model_path): os.makedirs(self.model_path)
		if not os.path.exists(self.log_path): os.makedirs(self.log_path)
		if not os.path.exists(self.colors_m): os.makedirs(self.colors_m)
		if not os.path.exists(self.depths_c1): os.makedirs(self.depths_c1)
		if not os.path.exists(self.depths_c2): os.makedirs(self.depths_c2)
		if not os.path.exists(self.colors_c1): os.makedirs(self.colors_c1)
		if not os.path.exists(self.colors_c2): os.makedirs(self.colors_c2)
		if not os.path.exists(self.depths_f1): os.makedirs(self.depths_f1)
		if not os.path.exists(self.colors_f1): os.makedirs(self.colors_f1)
		if not os.path.exists(self.colors_f2): os.makedirs(self.colors_f2)

	def get_near_far_earth(self, rays_o, rays_d, scene_scale):
		global_center = np.array([[0.0, 0.0, -6371011.0]], dtype=np.float32) * scene_scale
		earth_rad, earth_full = 6371011 * scene_scale, (6371011 + 250) * scene_scale

		a = np.sum(np.square(rays_d), axis=-1)
		b = 2. * np.sum((rays_o - global_center) * rays_d, axis=-1)

		c_near = np.sum(np.square(rays_o - global_center), axis=-1) - np.square(earth_full)
		c_far = np.sum(np.square(rays_o - global_center), axis=-1) - np.square(earth_rad)

		delta_near = np.sqrt(np.square(b) - 4. * a * c_near)
		delta_far = np.sqrt(np.square(b) - 4. * a * c_far)

		d_near = (-1. * b - delta_near) / (2. * a)
		d_far = (-1. * b - delta_far) / (2. * a)

		t_near = np.expand_dims(d_near * 0.9, axis=-1)
		t_far = np.expand_dims(d_far * 1.1, axis=-1)

		return t_near, t_far

	def get_near_far_planes(self, rays_o, rays_d, z_near, z_far, max_len):
		high2near = np.maximum(rays_o[0, 2] - z_near, 0.0)
		high2far = np.maximum(rays_o[0, 2] - z_far, 0.0)

		t_near = high2near / (np.abs(rays_d[:, 2]) + 1e-6)
		t_far = high2far / (np.abs(rays_d[:, 2]) + 1e-6)
		t_far = np.minimum(t_near + max_len, t_far)

		t_near = np.expand_dims(t_near, axis=-1)
		t_far = np.expand_dims(t_far, axis=-1)

		return t_near, t_far

	def get_bounding_box(self, poses, high, width, focal, scene_scale, z_near, z_far, max_len):
		coords = np.array([[0, 0], [0, width-1], [high-1, 0], [high-1, width-1]], dtype=np.float32) + 0.5

		xs = (coords[:, 1] - 0.5 * width) / focal
		ys = (coords[:, 0] - 0.5 * high) / -focal
		zs = np.ones_like(xs, dtype=np.float32) * -1.0

		directions = np.stack([xs, ys, zs], axis=-1)
		directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

		box_min = np.array([[100, 100, 100]], dtype=np.float32)
		box_max = np.array([[-100, -100, -100]], dtype=np.float32)

		for pose in poses:
			rays_o = np.tile(np.expand_dims(pose[:, 3], axis=0), [4, 1])
			rays_d = np.sum(np.expand_dims(directions, axis=1) * np.expand_dims(pose[:, :3], axis=0), axis=-1)
			rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

			if self.mode == "earth":
				t_near, t_far = self.get_near_far_earth(rays_o, rays_d, scene_scale)
			elif self.mode == "planes":
				t_near, t_far = self.get_near_far_planes(rays_o, rays_d, z_near, z_far, max_len)
			else:
				raise ValueError("Note: mode must be in [earth, planes]!")

			points_near = rays_o + t_near * rays_d
			points_far = rays_o + t_far * rays_d
			points = np.concatenate([points_near, points_far], axis=0)

			xyz_min = np.min(points, axis=0)
			xyz_max = np.max(points, axis=0)

			for i in range(3):
				if box_min[0, i] > xyz_min[i]: box_min[0, i] = xyz_min[i]
				if box_max[0, i] < xyz_max[i]: box_max[0, i] = xyz_max[i]

		return box_min, box_max

	def get_bounding_box_blender(self, high, width, focal, poses, p_near, p_far):
		coords = np.array([[0, 0], [0, width-1], [high-1, 0], [high-1, width-1]], dtype=np.float32) + 0.5

		xs = (coords[:, 1] - 0.5 * width) / focal
		ys = (coords[:, 0] - 0.5 * high) / -focal
		zs = np.ones_like(xs, dtype=np.float32) * -1.0

		directions = np.stack([xs, ys, zs], axis=-1)
		directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
		positions = []

		for pose in poses:
			rays_o = np.expand_dims(pose[:, 3], axis=0)
			rays_d = np.sum(np.expand_dims(directions, axis=1) * np.expand_dims(pose[:, :3], axis=0), axis=-1)
			rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
			pos_min = rays_o + p_near * rays_d
			pos_max = rays_o + p_far * rays_d
			positions.append(np.concatenate([pos_min, pos_max], axis=0))

		positions = np.concatenate(positions, axis=0)

		box_min = np.min(positions, axis=0, keepdims=True)
		box_max = np.max(positions, axis=0, keepdims=True)

		return box_min, box_max

	def load_dict_scene(self, data_dict):
		data_dict = data_dict

		with open(self.json_path, "r") as f:
			metadata = json.load(f)

		high, width, focal = metadata["high"], metadata["width"], metadata["focal"]

		data_dict.update({"high": high, "width": width, "focal": focal})

		if self.mode == "earth":
			scene_scale = metadata["scene_scale"]
			z_near, z_far, max_len = -0.9, -3.5, 30.0
		elif self.mode == "planes":
			scene_scale = 0.004656
			z_near, z_far, max_len = metadata["z_near"], metadata["z_far"], metadata["max_len"]
		else:
			raise ValueError("Note: mode must be in [earth, planes]!")

		data_dict.update({"scene_scale": scene_scale, "z_near": z_near, "z_far": z_far, "max_len": max_len})

		names = metadata["names"] if "names" in metadata.keys() else sorted(os.listdir(self.image_path))
		poses = [np.reshape(np.array(pose[:15]), [3, 5])[:, :4] for pose in metadata["poses"]]
		assert len(names) == len(poses), "Note: Numbers of image and pose must be matched."
		data_dict.update({"names": names, "poses": poses})

		if self.trainable:
			if os.path.exists(self.box_path):
				with open(self.box_path, "r") as f:
					data = json.load(f)

				box_min = np.array(data["box_min"], dtype=np.float32)
				box_max = np.array(data["box_max"], dtype=np.float32)

			else:
				box_min, box_max = self.get_bounding_box(poses, high, width, focal, scene_scale, z_near, z_far, max_len)
				data = {"box_min": box_min.tolist(), "box_max": box_max.tolist()}

				with open(self.box_path, "w") as f:
					json.dump(data, f, indent=4, ensure_ascii=False)

		else:
			if not os.path.exists(self.box_path):
				raise ValueError("Not found the file: " + self.box_path)

			with open(self.box_path, "r") as f:
				data = json.load(f)

			box_min = np.array(data["box_min"], dtype=np.float32)
			box_max = np.array(data["box_max"], dtype=np.float32)

		data_dict.update({"box_min": box_min, "box_max": box_max})

		return data_dict

	def load_dict_blender(self, data_dict):
		data_dict = data_dict

		with open(self.json_path, "r") as f:
			metadata = json.load(f)

		p_near, p_far, names, poses = metadata["p_near"], metadata["p_far"], [], []
		name = os.path.join(self.image_path, metadata["frames"][0]["file_path"].split("/")[-1]+".png")
		image = cv2.imread(name)
		high, width, _ = np.shape(image)
		focal = 0.5 * width / np.tan(0.5 * metadata["camera_angle_x"])

		data_dict.update({"high": high, "width": width, "focal": focal, "p_near": p_near, "p_far": p_far})

		for data in metadata["frames"]:
			names.append(data["file_path"].split("/")[-1]+".png")
			poses.append(np.array(data["transform_matrix"])[:3, :4])

		data_dict.update({"names": names, "poses": poses})

		if self.trainable:
			if not os.path.exists(self.box_path):
				box_min, box_max = self.get_bounding_box_blender(high, width, focal, poses, p_near, p_far)
				data = {"box_min": box_min.tolist(), "box_max": box_max.tolist()}

				with open(self.box_path, "w") as f:
					json.dump(data, f, indent=4, ensure_ascii=False)

			else:
				with open(self.box_path, "r") as f:
					data = json.load(f)

				box_min = np.array(data["box_min"], dtype=np.float32)
				box_max = np.array(data["box_max"], dtype=np.float32)

		else:
			if not os.path.exists(self.box_path):
				raise ValueError("Not found the file: " + self.box_path)

			with open(self.box_path, "r") as f:
				data = json.load(f)

			box_min = np.array(data["box_min"], dtype=np.float32)
			box_max = np.array(data["box_max"], dtype=np.float32)

		data_dict.update({"box_min": box_min, "box_max": box_max})

		return data_dict

	def load_data_dict(self):
		data_dict = {"nerf": False, "cnn": False}

		data_dict.update({"num_rays": self.num_rays, "pos_levels": self.pos_levels, "poi_levels": self.poi_levels, 
			"dir_levels": self.dir_levels, "appearance": self.appearance, "app_dims": self.app_dims})

		data_dict.update({"inters_proposal": self.inters_proposal, "inters_property": self.inters_property, 
			"batch_size": self.batch_size, "grid_x": self.grid_x, "grid_y": self.grid_y})

		data_dict.update({"units_ngp": self.units_ngp, "units_nerf": self.units_nerf, "units_dec": self.units_dec, 
			"units_ger": self.units_ger, "units_mlp": self.units_mlp, "units_cnn": self.units_cnn})

		data_dict.update({"hash_base": self.hash_base, "hash_finest": self.hash_finest, "hash_scale": self.hash_scale, 
			"hash_levels": self.hash_levels, "hash_size": self.hash_size})

		if self.mode == "blender":
			data_dict = self.load_dict_blender(data_dict)
		elif self.mode == "earth":
			data_dict = self.load_dict_scene(data_dict)
		elif self.mode == "planes":
			data_dict = self.load_dict_scene(data_dict)
		else:
			raise ValueError("Note: mode must be in [earth, planes, blender].")

		if self.trainable and self.stage == 0:
			dataset_train = datasets.DataSet_Stage0(self.image_path, data_dict["names"], data_dict["poses"], 
				data_dict["high"], data_dict["width"], self.num_rays)
			data_train, num_train = dataset_train.get_iter_data()
			epochs = self.max_rays // self.num_rays // num_train
			decay_rate = np.power(0.1, 1.0 / epochs)
			data_dict.update({"data_train": data_train, "num_poses": num_train, "epochs": epochs, "decay_rate": decay_rate})
			data_dict["nerf"] = True

		if self.trainable and self.stage > 0:
			dataset_train = datasets.DataSet_Stage1(self.image_path, self.colors_m, self.batch_size)
			data_train, num_train = dataset_train.get_iter_data()
			image_epochs = self.max_images // num_train
			decay_rate = np.power(0.1, 1.0 / image_epochs)
			num_iters = num_train // self.batch_size
			if num_iters * self.batch_size != num_train: num_iters += 1
			data_dict.update({"data_train": data_train, "num_poses": num_train, "image_epochs": image_epochs, 
				"decay_rate": decay_rate, "num_iters": num_iters})
			data_dict["cnn"] = True

		if not self.trainable:
			dataset_train = datasets.DataSet_Test(self.image_path, data_dict["names"], data_dict["poses"], 
				data_dict["high"], data_dict["width"], self.num_rays)
			data_train, num_train = dataset_train.get_iter_data()
			epochs = self.max_rays_test // self.num_rays // num_train
			decay_rate = np.power(0.1, 1.0 / epochs)
			data_dict.update({"data_train": data_train, "num_poses": num_train, "epochs": epochs, "decay_rate": decay_rate})

		return data_dict

	def save_configs(self):
		content = "data: %s\nmode: %s\ngrid_x: %d\ngrid_y: %d\n" % (self.data, self.mode, self.grid_x, self.grid_y)
		content += "max_rays: %d\nmax_images: %d\nmax_rays_test: %d\n" % (self.max_rays, self.max_images, self.max_rays_test)
		content += "inters_proposal: %d\ninters_property: %d\n" % (self.inters_proposal, self.inters_property)
		content += "num_rays: %d\nbatch_size: %d\napp_dims: %d\n" % (self.num_rays, self.batch_size, self.app_dims)
		content += "lr_nerf: %f\nlr_cnn: %f\nunits_ngp: %d\n" % (self.lr_nerf, self.lr_cnn, self.units_ngp)
		content += "units_nerf: %d\nunits_dec: %d\nunits_ger: %d\n" % (self.units_nerf, self.units_dec, self.units_ger)
		content += "units_mlp: %d\nunits_cnn: %d\npos_levels: %d\n" % (self.units_mlp, self.units_cnn, self.pos_levels)
		content += "poi_levels: %d\ndir_levels: %d\nstride_stage1: %d\n" % (self.poi_levels, self.dir_levels, self.stride_stage1)
		content += "hash_levels: %d\nhash_size: %d\nhash_base: %d\n" % (self.hash_levels, self.hash_size, self.hash_base)

		if self.hash_finest is not None:
			content += "hash_finest: %d\nhash_scale: None\n" % (self.hash_finest)
		else:
			content += "hash_finest: None\nhash_scale: %f\n" % (self.hash_scale)

		if self.trainable and self.stage == 0:
			with open(self.config_path, "w") as f:
				f.write(content)
			print("Write configs to: " + self.config_path)

	def prepare_for_render(self):
		self.high = self.data_dict["high"]
		self.width = self.data_dict["width"]
		self.pixels = self.high * self.width
		w_list, h_list = np.meshgrid(np.arange(self.width), np.arange(self.high))
		self.hw_list = np.reshape(np.stack([h_list, w_list], axis=-1), (self.pixels, 2))
		self.num_iters = self.pixels // self.num_rays
		if self.num_iters * self.num_rays != self.pixels: self.num_iters += 1

	def network_variables_initializer(self):
		def weights_init(inputs):
			if isinstance(inputs, nn.Linear):
				nn.init.xavier_uniform_(inputs.weight)
				if inputs.bias is not None:
					nn.init.zeros_(inputs.bias)
			elif isinstance(inputs, nn.Conv2d):
				nn.init.xavier_uniform_(inputs.weight)
				if inputs.bias is not None:
					nn.init.zeros_(inputs.bias)

		self.network.apply(weights_init)

	def get_parameters_number(self):
		total_num = sum(p.numel() for p in self.network.parameters())
		trainable_num = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
		notrainable_num = total_num - trainable_num

		print(">Network parameters: %d\n>Trainable parameters: %d\n>Notrainable parameters: %d\n" % (total_num, 
			trainable_num, notrainable_num))

		if hasattr(self, "app_embeddings"):
			total_num = sum(p.numel() for p in self.app_embeddings.parameters())
			trainable_num = sum(p.numel() for p in self.app_embeddings.parameters() if p.requires_grad)
			notrainable_num = total_num - trainable_num

			print(">Embedding parameters: %d\n>Trainable parameters: %d\n>Notrainable parameters: %d\n" % (
				total_num, trainable_num, notrainable_num))

	def load_weights(self):
		if os.path.exists(self.nerf_model):
			self.network.load_state_dict(torch.load(self.nerf_model), strict=False)
			print("Load network: " + self.nerf_model)

		if os.path.exists(self.cnn_model):
			self.network.load_state_dict(torch.load(self.cnn_model), strict=False)
			print("Load network: " + self.cnn_model)

	def forward(self, pose, focal, app_encodings):
		depths_c1 = np.zeros((self.high, self.width), dtype=np.float32)
		depths_c2 = np.zeros((self.high, self.width), dtype=np.float32)
		depths_f1 = np.zeros((self.high, self.width), dtype=np.float32)
		colors_c1 = np.zeros((self.high, self.width, 3), dtype=np.float32)
		colors_c2 = np.zeros((self.high, self.width, 3), dtype=np.float32)
		colors_f1 = np.zeros((self.high, self.width, 3), dtype=np.float32)
		start, stop = 0, self.num_rays

		for _ in range(self.num_iters):
			if stop > self.pixels:
				coords = np.concatenate([self.hw_list[start:], self.hw_list[:(stop-self.pixels)]], axis=0)
			else:
				coords = self.hw_list[start:stop]

			coords = torch.tensor(coords, dtype=torch.int32, device=device)
			results = self.network.forward_stage0(coords, pose, focal, app_encodings)
			depth_c1, depth_c2, color_c1, color_c2, depth_f1, color_f1, _ = results

			color_f1 = torch.clamp(color_f1, min=0.0, max=1.0)
			color_f1 = torch.nan_to_num(color_f1, nan=0.0, posinf=1.0, neginf=0.0)

			coords = coords.cpu().detach().numpy()
			depth_c1 = depth_c1.cpu().detach().numpy()
			depth_c2 = depth_c2.cpu().detach().numpy()
			color_c1 = color_c1.cpu().detach().numpy()
			color_c2 = color_c2.cpu().detach().numpy()
			depth_f1 = depth_f1.cpu().detach().numpy()
			color_f1 = color_f1.cpu().detach().numpy()

			depths_c1[coords[:, 0], coords[:, 1]] = depth_c1
			depths_c2[coords[:, 0], coords[:, 1]] = depth_c2
			colors_c1[coords[:, 0], coords[:, 1]] = color_c1
			colors_c2[coords[:, 0], coords[:, 1]] = color_c2
			depths_f1[coords[:, 0], coords[:, 1]] = depth_f1
			colors_f1[coords[:, 0], coords[:, 1]] = color_f1

			start += self.num_rays
			stop += self.num_rays

		colors_f1 = ((colors_f1 * 255.).astype(np.uint8) / 255.).astype(np.float32)
		colors_f1 = torch.unsqueeze(torch.tensor(colors_f1, dtype=torch.float32, device=device), dim=0)

		colors_f2 = self.network.forward_stage1(colors_f1)

		colors_f1 = torch.squeeze(colors_f1, dim=0).cpu().detach().numpy()
		colors_f2 = torch.squeeze(colors_f2, dim=0).cpu().detach().numpy()

		return depths_c1, depths_c2, colors_c1, colors_c2, depths_f1, colors_f1, colors_f2

	def prepare_for_stage1(self):
		assert self.trainable and self.stage == 0, "trainable must be True, and stage must be 0."

		self.network.hash_encoder.requires_grad_(False)
		self.network.get_inters_property.requires_grad_(False)
		self.network.depths_correct.requires_grad_(False)
		self.network.get_range.requires_grad_(False)
		self.network.get_fines_property.requires_grad_(False)

		if self.appearance:
			if hasattr(self, "app_embeddings"):
				self.app_embeddings.embeddings.requires_grad_(False)
			else:
				self.app_embeddings = mlp_utils.App_Embeddings(self.data_dict["num_poses"], self.app_dims)
				assert os.path.exists(self.app_train), "stage 0 may not be run, app_train is not found!"
				self.app_embeddings.load_state_dict(torch.load(self.app_train))
				print("Load network: " + self.app_train)
				self.app_embeddings.embeddings.requires_grad_(False)

		self.get_parameters_number()
		print("\nWrite data for stage1 to training ...")

		focal = torch.tensor(self.data_dict["focal"], dtype=torch.float32, device=device)

		for i in range(0, self.data_dict["num_poses"], self.stride_stage1):
			since_time, name = time.time(), self.data_dict["names"][i]
			pose = torch.tensor(self.data_dict["poses"][i], dtype=torch.float32, device=device)

			if self.appearance:
				app_encodings = self.app_embeddings(torch.tensor([i], dtype=torch.int32, device=device))
			else:
				app_encodings = None

			_, _, _, _, _, colors_f1, _ = self.forward(pose, focal, app_encodings)

			colors_f1 = (colors_f1 * 255.).astype(np.uint8)
			cv2.imwrite(os.path.join(self.colors_m, name), colors_f1[..., (2, 1, 0)])
			run_time = time.time() - since_time

			print("The %dth (tol: %d) results generate: %s  time: %.6fs" % (i, self.data_dict["num_poses"]-1, name, run_time))

	def train_nerf_module(self):
		if self.appearance:
			self.app_embeddings = mlp_utils.App_Embeddings(self.data_dict["num_poses"], self.app_dims)

			if os.path.exists(self.app_train):
				self.app_embeddings.load_state_dict(torch.load(self.app_train))
				print("Load network: " + self.app_train)

			self.app_embeddings.embeddings.requires_grad_(True)

		self.save_configs()
		self.get_parameters_number()

		init_lr = self.lr_nerf * np.power(self.data_dict["decay_rate"], self.init_epoch)

		if self.appearance:
			optimizer = torch.optim.Adam([{"params": self.network.parameters()}, 
				{"params": self.app_embeddings.parameters()}], lr=init_lr)
		else:
			optimizer = torch.optim.Adam(self.network.parameters(), lr=init_lr)

		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.data_dict["decay_rate"])
		save_epochs = self.data_dict["epochs"] // self.saves_min
		summarywriter = SummaryWriter(log_dir=self.log_path)
		focal = torch.tensor(self.data_dict["focal"], dtype=torch.float32, device=device)

		print("Training start (stage%d), images: %d  num_rays: %d  epochs: %d" % (self.stage, 
			self.data_dict["num_poses"], self.num_rays, self.data_dict["epochs"]))

		for epoch in range(self.init_epoch, self.data_dict["epochs"]):
			tol, l1, l2, hash1, hash2, dels, dist, effects, since_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, time.time()

			for _ in range(self.data_dict["num_poses"]):
				coords, pose, index, image = next(self.data_dict["data_train"])

				coords = torch.tensor(coords, dtype=torch.int32, device=device)
				pose = torch.tensor(pose, dtype=torch.float32, device=device)
				index = torch.tensor([index], dtype=torch.int32, device=device)
				image = torch.tensor(image, dtype=torch.float32, device=device)

				if self.appearance:
					app_encodings = self.app_embeddings(index)
				else:
					app_encodings = None

				results = self.network.forward_stage0(coords, pose, focal, app_encodings)
				_, depths_c2, colors_c1, colors_c2, depths_f1, colors_f1, distor_value = results

				depth_mask = self.depth_mask(coords)
				colors_gt = self.get_colors(coords, image)
				distor_gt = torch.zeros_like(distor_value, dtype=torch.float32, device=device)

				loss_l1 = self.loss_l1(colors_f1, colors_gt)
				loss_l2 = self.loss_l2(colors_f1, colors_gt)
				loss_hash1 = self.loss_l2(colors_c1, colors_gt)
				loss_hash2 = self.loss_l2(colors_c2, colors_gt)
				loss_depth = self.loss_l2(torch.mul(depths_f1, depth_mask), torch.mul(depths_c2, depth_mask))
				loss_dist = self.loss_l1(distor_value, distor_gt)

				loss_tol = 0.1 * loss_l1 + loss_l2 + 0.1 * loss_hash1 + loss_hash2 + loss_depth + 0.001 * loss_dist

				if torch.isnan(loss_tol):
					continue
				else:
					optimizer.zero_grad()
					loss_tol.backward()
					optimizer.step()
					effects += 1
					tol += loss_tol.item()
					l1 += loss_l1.item()
					l2 += loss_l2.item()
					hash1 += loss_hash1.item()
					hash2 += loss_hash2.item()
					dels += loss_depth.item()
					dist += loss_dist.item()

			if effects > 0:
				tol = tol / effects
				l1 = l1 / effects
				l2 = l2 / effects
				hash1 = hash1 / effects
				hash2 = hash2 / effects
				dels = dels / effects
				dist = dist / effects

			summarywriter.add_scalar("loss_tol", tol, global_step=epoch)
			summarywriter.add_scalar("loss_l1", l1, global_step=epoch)
			summarywriter.add_scalar("loss_l2", l2, global_step=epoch)
			summarywriter.add_scalar("loss_hash1", hash1, global_step=epoch)
			summarywriter.add_scalar("loss_hash2", hash2, global_step=epoch)
			summarywriter.add_scalar("loss_depth", dels, global_step=epoch)
			summarywriter.add_scalar("loss_dist", dist, global_step=epoch)
			summarywriter.add_scalar("learning_rate", optimizer.state_dict()["param_groups"][0]["lr"], global_step=epoch)
			scheduler.step()
			run_time = time.time() - since_time

			content0 = "\n>>>epoch: %d/%d  loss_tol: %.6f  " % (epoch + 1, self.data_dict["epochs"], tol)
			content1 = "loss_l1: %.6f  loss_l2: %.6f  loss_hash1: %.6f  loss_hash2: %.6f  " % (l1, l2, hash1, hash2)
			content2 = "loss_depth: %.6f  loss_dist: %.6f  effects: %d  time: %.6fs" % (dels, dist, effects, run_time)
			print(content0 + content1 + content2)

			if (epoch + 1) % save_epochs == 0:
				torch.save(self.network.state_dict(), self.nerf_model)
				print("\nWrite network: " + self.nerf_model)

				if self.appearance:
					torch.save(self.app_embeddings.state_dict(), self.app_train)
					print("Write network: " + self.app_train)

		torch.save(self.network.state_dict(), self.nerf_model)
		print("\nWrite network: " + self.nerf_model)

		if self.appearance:
			torch.save(self.app_embeddings.state_dict(), self.app_train)
			print("Write network: " + self.app_train)

		self.prepare_for_stage1()
		print("\nStage0 training end!")

	def train_cnn_module(self):
		self.get_parameters_number()

		init_lr = self.lr_cnn * np.power(self.data_dict["decay_rate"], self.init_epoch)
		optimizer = torch.optim.Adam(self.network.parameters(), lr=init_lr)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.data_dict["decay_rate"])
		save_epochs = self.data_dict["image_epochs"] // self.saves_min
		summarywriter = SummaryWriter(log_dir=self.log_path)

		print("Training start (stage%d), images: %d  batch_size: %d  epochs: %d" % (self.stage, 
			self.data_dict["num_poses"], self.batch_size, self.data_dict["image_epochs"]))

		for epoch in range(self.init_epoch, self.data_dict["image_epochs"]):
			tol, l1, l2, since_time = 0.0, 0.0, 0.0, time.time()

			for _ in range(self.data_dict["num_iters"]):
				colors_f1, colors_gt = next(self.data_dict["data_train"])

				colors_f1 = torch.tensor(colors_f1, dtype=torch.float32, device=device)
				colors_gt = torch.tensor(colors_gt, dtype=torch.float32, device=device)

				colors_f2 = self.network.forward_stage1(colors_f1)

				loss_l1 = self.loss_l1(colors_f2, colors_gt)
				loss_l2 = self.loss_l2(colors_f2, colors_gt)

				loss_tol = 0.1 * loss_l1 + loss_l2

				optimizer.zero_grad()
				loss_tol.backward()
				optimizer.step()

				tol += loss_tol.item()
				l1 += loss_l1.item()
				l2 += loss_l2.item()

			tol = tol / self.data_dict["num_iters"]
			l1 = l1 / self.data_dict["num_iters"]
			l2 = l2 / self.data_dict["num_iters"]

			summarywriter.add_scalar("loss_tol", tol, global_step=epoch)
			summarywriter.add_scalar("loss_l1", l1, global_step=epoch)
			summarywriter.add_scalar("loss_l2", l2, global_step=epoch)
			summarywriter.add_scalar("learning_rate", optimizer.state_dict()["param_groups"][0]["lr"], global_step=epoch)
			scheduler.step()
			run_time = time.time() - since_time

			content0 = "\n>>>epoch: %d/%d  loss_tol: %.6f  " % (epoch + 1, self.data_dict["image_epochs"], tol)
			content1 = "loss_l1: %.6f  loss_l2: %.6f  time: %.6fs" % (l1, l2, run_time)
			print(content0 + content1)

			if (epoch + 1) % save_epochs == 0:
				torch.save(self.network.state_dict(), self.cnn_model)
				print("\nWrite network: " + self.cnn_model)

		torch.save(self.network.state_dict(), self.cnn_model)
		print("\nWrite network: " + self.cnn_model + "\nStage1 training end!")

	def train(self):
		if not self.trainable: raise ValueError("trainable must be True in training mode!")

		if self.stage == 0:
			self.train_nerf_module()
		else:
			self.train_cnn_module()

	def rende_train(self, start=0, stride=1):
		assert self.trainable and self.stage == 0, "rende training dataset must be set trainable and stage to True and 0"

		self.network.hash_encoder.requires_grad_(False)
		self.network.get_inters_property.requires_grad_(False)
		self.network.depths_correct.requires_grad_(False)
		self.network.get_range.requires_grad_(False)
		self.network.get_fines_property.requires_grad_(False)
		self.network.render_refinement.requires_grad_(False)

		if self.appearance:
			self.app_embeddings = mlp_utils.App_Embeddings(self.data_dict["num_poses"], self.app_dims)
			assert os.path.exists(self.app_train), "training is not finshed, app_train is not found!"
			self.app_embeddings.load_state_dict(torch.load(self.app_train))
			print("Load network: " + self.app_train)
			self.app_embeddings.embeddings.requires_grad_(False)

		self.get_parameters_number()

		names, poses, num_poses = self.data_dict["names"], self.data_dict["poses"], self.data_dict["num_poses"]
		focal = torch.tensor(self.data_dict["focal"], dtype=torch.float32, device=device)
		assert start < num_poses, "start must be less than the number of poses: %d" % (num_poses)

		for i in range(start, num_poses, stride):
			since_time = time.time()
			pose = torch.tensor(poses[i], dtype=torch.float32, device=device)

			if self.appearance:
				app_encodings = self.app_embeddings(torch.tensor([i], dtype=torch.int32, device=device))
			else:
				app_encodings = None

			results = self.forward(pose, focal, app_encodings)
			depths_c1, depths_c2, colors_c1, colors_c2, depths_f1, colors_f1, colors_f2 = results

			depths_c1 = post_display(depths_c1)
			depths_c2 = post_display(depths_c2)
			depths_f1 = post_display(depths_f1)
			colors_c1 = (colors_c1 * 255.).astype(np.uint8)
			colors_c2 = (colors_c2 * 255.).astype(np.uint8)
			colors_f1 = (colors_f1 * 255.).astype(np.uint8)
			colors_f2 = (colors_f2 * 255.).astype(np.uint8)
			run_time = time.time() - since_time

			cv2.imwrite(os.path.join(self.depths_c1, names[i]), depths_c1[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.depths_c2, names[i]), depths_c2[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.depths_f1, names[i]), depths_f1[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.colors_c1, names[i]), colors_c1[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.colors_c2, names[i]), colors_c2[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.colors_f1, names[i]), colors_f1[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.colors_f2, names[i]), colors_f2[..., (2, 1, 0)])

			print("The %dth (tol: %d) results generated: %s  time: %.6fs" % (i, num_poses-1, names[i], run_time))

	def embeddings_test_init(self):
		data = torch.load(self.app_train)
		embeddings = data["embeddings.weight"]
		train_num = embeddings.shape[0]
		indic = [i for i in range(3, train_num, 4)]
		if len(indic) != self.data_dict["num_poses"]: indic = indic[:-1]
		indic = torch.tensor(indic, dtype=torch.int64, device=device)
		data["embeddings.weight"] = embeddings[indic]
		self.app_embeddings.load_state_dict(data)
		print("Initialization appearance embeddings for testing, complete.")

	def get_embeddings_test(self):
		assert (not self.trainable) and self.appearance, "trainable and appearance must be False and True in testing"
		self.app_embeddings = mlp_utils.App_Embeddings(self.data_dict["num_poses"], self.app_dims)
		self.embeddings_test_init()

		if os.path.exists(self.app_test):
			self.app_embeddings.load_state_dict(torch.load(self.app_test))
			print("Load network: " + self.app_test)

		self.app_embeddings.embeddings.requires_grad_(True)
		self.get_parameters_number()

		optimizer = torch.optim.Adam(self.app_embeddings.parameters(), lr=self.lr_nerf)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.data_dict["decay_rate"])
		summarywriter = SummaryWriter(log_dir=self.log_path)
		focal = torch.tensor(self.data_dict["focal"], dtype=torch.float32, device=device)

		for epoch in range(self.data_dict["epochs"]):
			tol, l1, l2, hash1, hash2, effects, since_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0, time.time()

			for _ in range(self.data_dict["num_poses"]):
				coords, pose, index, image = next(self.data_dict["data_train"])

				coords = torch.tensor(coords, dtype=torch.int32, device=device)
				pose = torch.tensor(pose, dtype=torch.float32, device=device)
				index = torch.tensor([index], dtype=torch.int32, device=device)
				image = torch.tensor(image, dtype=torch.float32, device=device)

				app_encodings = self.app_embeddings(index)
				results = self.network.forward_stage0(coords, pose, focal, app_encodings)
				_, _, colors_c1, colors_c2, _, colors_f1, _ = results

				colors_gt = self.get_colors(coords, image)

				loss_l1 = self.loss_l1(colors_f1, colors_gt)
				loss_l2 = self.loss_l2(colors_f1, colors_gt)
				loss_hash1 = self.loss_l2(colors_c1, colors_gt)
				loss_hash2 = self.loss_l2(colors_c2, colors_gt)

				loss_tol = 0.1 * loss_l1 + loss_l2 + 0.1 * loss_hash1 + loss_hash2

				if torch.isnan(loss_tol):
					continue
				else:
					optimizer.zero_grad()
					loss_tol.backward()
					optimizer.step()
					effects += 1
					tol += loss_tol.item()
					l1 += loss_l1.item()
					l2 += loss_l2.item()
					hash1 += loss_hash1.item()
					hash2 += loss_hash2.item()

			if effects > 0:
				tol = tol / effects
				l1 = l1 / effects
				l2 = l2 / effects
				hash1 = hash1 / effects
				hash2 = hash2 / effects

			summarywriter.add_scalar("loss_tol", tol, global_step=epoch)
			summarywriter.add_scalar("loss_l1", l1, global_step=epoch)
			summarywriter.add_scalar("loss_l2", l2, global_step=epoch)
			summarywriter.add_scalar("loss_hash1", hash1, global_step=epoch)
			summarywriter.add_scalar("loss_hash2", hash2, global_step=epoch)
			summarywriter.add_scalar("learning_rate", optimizer.state_dict()["param_groups"][0]["lr"], global_step=epoch)
			scheduler.step()
			run_time = time.time() - since_time

			content0 = "\n>>>epoch: %d/%d  loss_tol: %.6f  loss_l1: %.6f  loss_l2: %.6f" % (epoch+1,self.data_dict["epochs"],tol,l1,l2)
			content1 = "  loss_hash1: %.6f  loss_hash2: %.6f  effects: %d  time: %.6fs" % (hash1, hash2, effects, run_time)
			print(content0 + content1)

		torch.save(self.app_embeddings.state_dict(), self.app_test)
		print("\nWrite network: " + self.app_test)

	def rende_test(self, start=0, stride=1):
		assert (not self.trainable), "trainable must be set to False!"
		assert start < self.data_dict["num_poses"], "start must be less than %d" % (self.data_dict["num_poses"])

		if self.appearance:
			self.app_embeddings = mlp_utils.App_Embeddings(self.data_dict["num_poses"], self.app_dims)
			assert os.path.exists(self.app_test), "get_embeddings_test may not be run, app_test is not found!"
			self.app_embeddings.load_state_dict(torch.load(self.app_test))
			print("Load network: " + self.app_test)
			self.app_embeddings.embeddings.requires_grad_(False)

		self.get_parameters_number()

		focal = torch.tensor(self.data_dict["focal"], dtype=torch.float32, device=device)
		poses, names, num_poses = self.data_dict["poses"], self.data_dict["names"], self.data_dict["num_poses"]

		for i in range(start, num_poses, stride):
			name, since_time = names[i], time.time()
			pose = torch.tensor(poses[i], dtype=torch.float32, device=device)

			if self.appearance:
				app_encodings = self.app_embeddings(torch.tensor([i], dtype=torch.int32, device=device))
			else:
				app_encodings = None

			results = self.forward(pose, focal, app_encodings)
			depths_c1, depths_c2, colors_c1, colors_c2, depths_f1, colors_f1, colors_f2 = results

			depths_c1 = post_display(depths_c1)
			depths_c2 = post_display(depths_c2)
			depths_f1 = post_display(depths_f1)
			colors_c1 = (colors_c1 * 255.).astype(np.uint8)
			colors_c2 = (colors_c2 * 255.).astype(np.uint8)
			colors_f1 = (colors_f1 * 255.).astype(np.uint8)
			colors_f2 = (colors_f2 * 255.).astype(np.uint8)
			run_time = time.time() - since_time

			cv2.imwrite(os.path.join(self.depths_c1, name), depths_c1[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.depths_c2, name), depths_c2[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.depths_f1, name), depths_f1[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.colors_c1, name), colors_c1[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.colors_c2, name), colors_c2[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.colors_f1, name), colors_f1[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(self.colors_f2, name), colors_f2[..., (2, 1, 0)])

			print("The %dth (tol: %d) results generated: %s  time: %.6fs" % (i, num_poses-1, name, run_time))

	def rende_json(self, json_path, app_path=None):
		assert (not self.trainable), "trainable must be set to False!"

		depths_path = self.results_path + "/depths_demo"
		colors_path = self.results_path + "/colors_demo"
		if not os.path.exists(depths_path): os.makedirs(depths_path)
		if not os.path.exists(colors_path): os.makedirs(colors_path)

		with open(json_path, "r") as f:
			metadata = json.load(f)

		focal = torch.tensor(self.data_dict["focal"], dtype=torch.float32, device=device)
		poses = [np.reshape(np.array(pose[:15]), (3, 5))[:, :4] for pose in metadata["poses"]]
		num_poses = len(poses)

		names = metadata["names"] if "names" in metadata.keys() else ["{:06d}.png".format(i) for i in range(num_poses)]

		if self.appearance:
			self.app_embeddings = mlp_utils.App_Embeddings(num_poses, self.app_dims)
			assert app_path is not None, "app_path cannot be None when appearance is set to True"
			self.app_embeddings.load_state_dict(torch.load(app_path))
			print("Load network: " + app_path)
			self.app_embeddings.embeddings.requires_grad_(False)

		self.get_parameters_number()

		for i in range(num_poses):
			since_time = time.time()
			name, pose = names[i], poses[i]
			pose = torch.tensor(pose, dtype=torch.float32, device=device)

			if self.appearance:
				app_encodings = self.app_embeddings(torch.tensor([i], dtype=torch.int32, device=device))
			else:
				app_encodings = None

			_, _, _, _, depths_f1, _, colors_f2 = self.forward(pose, focal, app_encodings)

			depths_f1 = post_display(depths_f1)
			colors_f2 = (colors_f2 * 255.).astype(np.uint8)
			run_time = time.time() - since_time

			cv2.imwrite(os.path.join(depths_path, name), depths_f1[..., (2, 1, 0)])
			cv2.imwrite(os.path.join(colors_path, name), colors_f2[..., (2, 1, 0)])
			print("The %dth (tol: %d) results generated: %s  time: %.6fs" % (i+1, num_poses, name, run_time))