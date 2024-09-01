# HS-Surf
HS-Surf: A Novel High-Frequency Surface Shell Radiance Field to Improve Large-Scale Scene Rendering

本项目的模型文件是 network.NetWork, 该模型会传入 mc_base.MCBase_NeRF 中进行训练和渲染测试.

该项目一共渲染六个场景, 分别是 data(Transamerica), google(56 Leonard), building, rubble, residence, campus, 它们的配置文件是config_xxx.py

****模型训练分为两个阶段，注意 MCBase_NeRF 的参数 trainable 指定当前是训练集还是测试集：

1. 第一阶段的训练，生成辐射场和高频外壳，快捷方式: python training_stage0.py, 详情见 training.py，需要在该文件中执行以下的代码(stage指定训练阶段，取值0或1)：

mcbase_nerf = MCBase_NeRF(configs=configs, trainable=True, rende_only=False, stage=0, init_epoch=0)

mcbase_nerf.train()

Note: mcbase_nerf.train() 会在训练结束后生成用于第二阶段的训练数据，如果出现意外，可以单独运行 mcbase_nerf.prepare_for_stage1()


2. 第二阶段的训练，生成为渲染结果去除噪声的CNN, 快捷方式: python training_stage1.py, 详情见training.py, 这不是一个必要的过程，但能进一步提升视觉质量，执行以下代码进行第二阶段的训练：

mcbase_nerf = MCBase_NeRF(configs=configs, trainable=True, rende_only=False, stage=1, init_epoch=0)

mcbase_nerf.train()


****场景的渲染请查看 testing.py，可以直接运行: python testing.py

如果每张图像都有自己的appearance embedding, 也就是config_xxx.py中的app_dims>0, 在渲染前需要计算每张测试图像的appearance embedding。需在testing.py中运行以下代码：

mcbase_nerf = MCBase_NeRF(configs=configs, trainable=False, rende_only=False, stage=0, init_epoch=0)

mcbase_nerf.get_embeddings_test()

最后进行渲染：

mcbase_nerf = MCBase_NeRF(configs=configs, trainable=False, rende_only=True, stage=1, init_epoch=0)

mcbase_nerf.rende_test()

如果没有为每张图像指定一个appearance embedding (config_xxx.py中的app_dims==0)，直接在测试数据上进行渲染：

mcbase_nerf = MCBase_NeRF(configs=configs, trainable=False, rende_only=True, stage=1, init_epoch=0)

mcbase_nerf.rende_test()


****我们提供了模型，训练和测试的全套代码， 并且提供了六个场景上已经训练好的checkpoint，方便其他研究者进行测试和比较。

****因为所有的checkpoint都是在3090GPU上训练得到，所以，建议它们的测试环境最好也是3090，否则可能会出现一些奇怪的现象。

****Building场景中已经发现的一种现象是3090上得到的checkpoint在3080上会生成奇怪的白色噪点。

****如果只有其它型号的GPU，可以重新跑一遍训练过程，这会消耗较长的时间。
