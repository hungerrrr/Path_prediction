# 方案 A：EigenTrajectory Transformer

本目录包含方案 A 的训练、预测、模型定义和数据工具，不导入 GRU 或方案 B 目录。

    python src\eigen_transformer\train.py --dataset-dir data\processed --model-out models\eigen_transformer.pt
    python src\eigen_transformer\predict.py --keypoints examples\inputs\urfall_keypoints_adl-01-cam0.xlsx --model models\eigen_transformer.pt --output-dir outputs\transformer
