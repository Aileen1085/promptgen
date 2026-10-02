# VISTA3D + 3D PromptGen（AMOS CT）

本目录是独立的 AMOS CT 实验入口，不修改或恢复正在运行的 SAM2/v10.2 训练。PromptGen 保留原有三维 token、三维 dense prompt 与三医学窗输入；专属 adapter 把它们映射到冻结的 VISTA3D research point-head transformer。VISTA3D 的公开 `forward` 只接收点击/类别编号，**不会原生接收 PromptGen 的 dense embedding**，因此本实验使用官方 `image_encoder` 和 `point_head` 的张量边界，不宣称是官方原样的 click 推理。AMOS 全局类别 ID 不直接传给 VISTA 类别头，避免两套 ID 语义误配。

## 固定协议

- 与 `segvol-3d/` 读取同一 `configs/ct13_v10_2_split_20260928.json` 和 `configs/ct13_v10_2_validation_four_new_sources4_classweight_v3_20260928.json`，仅选 AMOS CT：200 train case、100 held-out case、45 个固定 case-class 验证任务；MRI 排除。入口校验划分、配对、类别映射、训练/验证无交集，并保存两个 JSON 的 SHA-256。
- 上述 JSON 的批准 SHA-256 分别固定为 `99b2eee8b4ae549537922614b0570665c6fa90387e21e04c69c04ed0652de022` 与 `d02bde8c7de9b9a20672df2cdc0ca39775a591eef33b89457cdac291c144dc39`；即使病例数相同，哈希不同也拒绝运行。
- 直接复用 `v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1` 中的 case 级 CT 缓存与 `.cache/prompt_planes` 的提示缓存，以及 v10.2 自适应 scribble 规则；不产生新的 CT 磁盘副本。VISTA 输入由已缓存的 canonical axial-first CT 在内存中按官方 CT 窗构造。
- 3D patch 默认 `96×96×96`（VISTA 排列 `R,A,Z`），深度步长 48。PromptGen 仍用 `Z,R,A`；所有轴交换写在 `data_bridge.py`。长目标完整 ROI 滑窗覆盖；完整 canonical 体积一次计分，ROI 外 GT 算 FN。
- 当前实现为保持已有 prompt/CT 网格严格同步，只在内存中缩放完整 XY 至 96，不执行官方参考验证配置中的 1.5 mm 各向同性重采样与前景局部裁剪。因此它是需要真实 AMOS 冒烟和固定验证检查的适配基线，**不能预设与官方 VISTA 预训练尺度完全对齐或达到其报告性能**。若后续加入物理尺度对齐，CT、GT、三窗视频、scribble 坐标与完整体积逆映射必须一起改，不能单独重采样 CT。
- 验证报告 Dice、IoU、precision、recall、specificity、pred/GT、3D NSD@1mm、3D HD95。该协议与 v10.2 的 `mean_slice_nsd_1mm` 不等价，不可直接比较 NSD。
- VISTA image encoder、point head、class head 冻结；只训练转入的 v10.2 PromptGen 与专属 feature/prompt adapter。v10.2 optimizer 不迁移。训练默认前 5 epoch 仅训练 adapter，其后联合微调 PromptGen；每 5 epoch 验证、连续 4 次无显著 Dice 提升早停。
- 原 PromptGen 视频只在内存中一次性缩到 checkpoint 的 work size（当前为 192），避免保留完整 ROI 的 1024² 视频；入口核对 work size。恢复时核对 VISTA 原始权重 SHA-256；`validate` 必须传入本版本已训练的 `--resume`，不会评估随机 adapter。训练与冒烟使用相同的 bf16 autocast。

## 源码与权重

使用 [官方 VISTA research 实现](https://github.com/Project-MONAI/VISTA/tree/main/vista3d)，`--vista-source` 指向包含 `vista3d/build_vista3d.py` 的源码目录。使用与 research 源码匹配的 MONAI 1.3 checkpoint `model_monai1.3.pt`，由官方 README 指向 `nvidia/NV-Segment-CT/vista3d_pretrained_model/model_monai1.3.pt`。不要拿 MONAI 1.4+ bundle checkpoint 混用：入口严格加载 `image_encoder`、`point_head`、`class_head` 全部键，发现不匹配即失败。

示例（在真实数据/权重所在环境中执行；先 audit、再 smoke，**不要直接开始正式训练**）：

本地工作副本目前没有上述两个审计 JSON，也没有对应 AMOS CT/GT；它们在原训练环境中必须保持原文件内容与哈希。缺失时入口会明确报错，**不要在本地重新随机划分**。

```bash
python vista3d/train.py audit-data
python vista3d/train.py smoke --vista-source /path/to/VISTA/vista3d \
  --vista-checkpoint /path/to/model_monai1.3.pt \
  --promptgen-checkpoint /path/to/v10_2_best.pth
python vista3d/train.py train --vista-source /path/to/VISTA/vista3d \
  --vista-checkpoint /path/to/model_monai1.3.pt \
  --promptgen-checkpoint /path/to/v10_2_best.pth
```

本地接口测试：`python -m unittest discover -s vista3d/tests -v`。本机尚未用官方 VISTA 权重、真实 AMOS 任务完成 GPU 冒烟；因此当前**只是可测试的实验实现，不是已验证性能或已启动训练**。正式运行前须检查官方源码/权重严格加载、真实 case loss/梯度、45 项完整验证覆盖与 GPU 显存。

## 2026-10-01 服务器验证记录

- 官方 research 源码稀疏检出于 `vista3d/official_source/vista3d`，提交 `d4a8fe0dbf5cb4b76c531fccca8e29c1e6f6ee45`；只检出模型和 scripts，未下载演示素材。
- 官方 MONAI 1.3 权重位于 `vista3d/official_weights/model_monai1.3.pt`，SHA-256 为 `889042ab37dbb9f9b2467e4a91654fb3b56482b74b98d315fc89026ab1570af8`，与发布页一致。隔离验证环境为 `vista3d/.venv-vista`，MONAI 1.3.2；未修改现有 `al` 环境。
- 服务器 37 项单元/接口测试通过；AMOS 审计为 200 个训练 CT、100 个留出 CT、45 个固定验证任务，复用 case CT 与自适应 scribble 缓存。官方模型约 2.18 亿参数严格加载成功且全部冻结。
- 在 GPU 均被其他作业占用时，用 CPU 和 `--depth 32 --stride 16 --vista-hw 32` 完成真实 AMOS 冒烟：loss 为有限值 `1.2927413`；PromptGen、feature bridge、prompt adapter 均有有限非零梯度；一个固定留出任务恢复到完整 `101×768×768` canonical 体积。该测试验证接线和数据/梯度通路，不代表模型性能；尚未测试默认 96³ GPU 显存，也未启动正式训练。
- 默认 `96×96×96` 输入的 CPU 冒烟在持续计算 20 分钟后触发预设超时（退出码 124），期间未输出异常；因此默认尺寸的完整前反向及 GPU 显存仍未验证，不能把缩小尺寸的结果外推为默认配置通过。
- 后续在物理 GPU2（UUID `GPU-06fff85b-357a-f4ed-8267-4229321acede`，本进程 `CUDA_VISIBLE_DEVICES=2` 后使用 `cuda:0`）完成默认 `96×96×96` 真实 AMOS GPU 冒烟，退出码 0、有限 loss `2.63184595`，PromptGen、feature bridge、prompt adapter 均有非零有限梯度，固定留出任务恢复至 `101×768×768` 完整体积。该卡原有 SegVol 进程保留，未终止或重启。
- 另在 tmux `vista3d_amos_gpu2_test_20261001` 开启独立两轮短程训练测试：每轮 2 个真实 case-class optimizer step，第二轮执行 45 项固定验证；日志 `vista3d/amos_gpu2_training_test_20261001.log`，独立输出 `vista3d/output/amos_gpu2_training_test_20261001`。第一轮两步及 `last.pth` 已产出，第二轮两步已完成且 loss 有限；验证结果尚未完成，不得将当前零/低 Dice 解释为正式训练性能。

## 2026-10-02 真正多尺度 FPN 对照

原 `VistaFeatureBridge` 从 VISTA decoder 的同一份 48 通道 point feature 投影出 `fpn0`、`fpn1`、`top`，三个名字不代表不同编码层。新增 `--feature-bridge-mode multiscale` 在一次冻结的官方 encoder 前向中取第 2/3/4 层：分别是原生 192 通道、1/4 分辨率；384 通道、1/8 分辨率；768 通道、1/16 分辨率。三个独立 1×1×1 卷积只将通道变成 PromptGen 期望的 32/64/256；各层的 XY 分辨率保留到 PromptGen 自己的融合处，轴向深度才插值对齐到同一帧数。旧 `single` 路径保持可选，checkpoint 带不同 architecture 标签，不能跨模式恢复。

两组均须从同一个 `v10/output/v10_2_ct13_e430_plateau_e510_20260930/20260930_152155/epoch470.pth` 只迁移 PromptGen 模型权重，不能恢复 v10.2 optimizer；使用同一官方 VISTA 权重、AMOS 划分、45 项固定验证、随机种子、3 医学窗、prompt/ROI、loss、96³ patch、stride 48、200 任务/epoch、40 epoch、每 5 轮验证、LR/早停和阈值 0.60。公共 3D prompt adapter 在两组随机种子相同时初始化完全一致；模型构建后重置 Python、NumPy 与 torch/CUDA 随机流，避免不同 bridge 参数量改变后续 dropout 序列；仅 feature bridge 不同。既有 E40 单特征 run 使用的 `best.pth` 与指定 `epoch470.pth` 的 PromptGen 参数不完全相同，因此**不得拿既有 E40 指标作为这次严格对照的单特征组**。

单特征组与多尺度组分别用独立输出目录，基于同协议验证比较 Dice、IoU、3D NSD@1mm、precision、recall、pred/GT、HD95 和各类别；同时检查显存、吞吐与是否早停。不能只凭训练 loss 或不同协议指标断言多尺度有效。
