# 方案 B：Leapfrog Diffusion

本目录包含方案 B 的训练、预测、模型定义和数据工具，不导入 GRU 或方案 A 目录。

    python src\leapfrog_diffusion\train.py --dataset-dir data\processed --model-out models\leapfrog_diffusion.pt
    python src\leapfrog_diffusion\predict.py --keypoints examples\inputs\urfall_keypoints_adl-01-cam0.xlsx --model models\leapfrog_diffusion.pt --output-dir outputs\diffusion
