from mc_base import MCBase_NeRF

if __name__ == "__main__":
	from config_data import get_configs
	configs = get_configs()
	mcbase_nerf = MCBase_NeRF(configs=configs, trainable=False, rende_only=True, stage=1, init_epoch=0)
	#mcbase_nerf.get_embeddings_test()
	mcbase_nerf.rende_test()