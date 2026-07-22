# src 代码结构

三套模型按目录完全分开，不会互相导入：

- gru/：原始 GRU。
- eigen_transformer/：方案 A，低秩多模态 Transformer。
- leapfrog_diffusion/：方案 B，Leapfrog 条件扩散。
- build_aligned_dataset.py、data_alignment.py：模型无关的原始数据对齐工具。
- evaluate_models.py：统一比较三套模型的可选评估工具。

每个模型目录都自带 core.py、prepare_dataset.py、train.py、predict.py 和视频可视化工具。A、B 还各自带有本地 common.py、models.py 和 prediction.py，运行时不会导入另外两个模型目录。

请从项目根目录使用 fall_detect 环境直接执行，例如：

    conda activate fall_detect
    python src\gru\train.py --help
    python src\eigen_transformer\train.py --help
    python src\leapfrog_diffusion\train.py --help
