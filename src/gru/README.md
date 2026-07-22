# 原始 GRU

- prepare_dataset.py：生成训练窗口。
- train.py：训练 GRU。
- predict.py：单轨迹预测及视频导出。
- core.py：GRU、数据读取、坐标标准化和检查点工具。

    python src\gru\train.py --dataset-dir data\processed --model-out models\trajectory_gru.pt
    python src\gru\predict.py --keypoints examples\inputs\urfall_keypoints_adl-01-cam0.xlsx --model models\trajectory_gru.pt --output-dir outputs\gru
