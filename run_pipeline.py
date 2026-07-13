import os
from config import *

def main():
    # 创建输出文件夹
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    print("===== 启动完整人体路径预测Pipeline =====")
    # 1. 骨架数据预处理
    from data_preprocess import preprocess_keypoint_data
    df_aligned, frame_map = preprocess_keypoint_data()
    # 2. 构建滑动窗口轨迹数据集
    from trajectory_dataset import build_trajectory_dataset
    traj_ds = build_trajectory_dataset(df_aligned)
    # 3. 卡尔曼平滑+批量路径预测
    from GRU_predictor_fixed import batch_predict
    pred_result = batch_predict(traj_ds)
    # 4. 转换为标准schema JSON
    from format_converter import convert_to_standard_schema
    convert_to_standard_schema(OUT_GRU_PRED, OUT_STANDARD_JSON)
    # 5. 生成可视化视频
    from video_visualizer_fixed import render_visual_video
    render_visual_video()

    print("\n===== Pipeline全部执行完成，输出文件清单 =====")
    print(f"1. 标准预测JSON（匹配需求schema）：{OUT_STANDARD_JSON}")
    print(f"2. 预测误差统计CSV：{OUT_ERROR_CSV}")
    print(f"3. 可视化预测视频：{OUT_VIS_VIDEO}")
    print("==============================================")

if __name__ == "__main__":
    main()