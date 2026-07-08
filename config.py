# ====================== 全局基础配置 ======================
# 视频参数
FPS = 30
VIDEO_NAME = "adl-01-cam0.mp4"
COORD_SYSTEM = "original_video_pixels"
# 观测/预测时长（可自由修改）
OBS_SECOND = 2.0
PRED_SECOND = 3.0
OBS_FRAMES = int(OBS_SECOND * FPS)
PRED_FRAMES = int(PRED_SECOND * FPS)
WINDOW_TOTAL_LEN = OBS_FRAMES + PRED_FRAMES

# RGB画面坐标范围（原始640*480 RGB）
RGB_X_OFFSET = 320
RGB_RAW_W = 640
RGB_RAW_H = 480
CLIP_X_MIN = 320
CLIP_X_MAX = 640
CLIP_Y_MIN = 0
CLIP_Y_MAX = 240

# 可视化拼接视频参数（左320深度，右320RGB）
TARGET_VIDEO_W = 640
TARGET_VIDEO_H = 240
W_SCALE = TARGET_VIDEO_H / RGB_RAW_H  # 0.5
H_SCALE = TARGET_VIDEO_H / RGB_RAW_H  # 0.5

# 骨架参与预测的关键点（头、髋、双脚，共4点）
KEYPOINTS_FOR_CENTER = [
    ("nose", "nosex", "nosey"),
    ("midhip", "midhipx", "midhipy"),
    ("rankle", "ranklex", "rankley"),
    ("lankle", "lanklex", "lankley"),
]
# 人体中心加权权重（髋部权重最高，双脚次之，头部辅助）
KP_WEIGHTS = [0.3, 0.5, 0.1, 0.1]

# # 走廊多边形配置（适配人体尺寸）
# CORRIDOR_PADDING_X = 25  # 水平方向padding（覆盖人体宽度）
# CORRIDOR_PADDING_Y = 40  # 垂直方向padding（覆盖人体高度）
# HUMAN_BBOX_W = 40        # 人体宽度（像素）
# HUMAN_BBOX_H = 120       # 人体高度（像素）

# BBox扩展像素（基于预测路径生成安全边界框的余量）
# 替代原「走廊多边形膨胀像素」，含义更清晰
# CORRIDOR_PADDING = 50

# 人体BBox适配配置（替代原固定CORRIDOR_PADDING）
HUMAN_BBOX_MARGIN = 8  # 人体真实BBox的扩展margin（仅8像素，贴合人体）
CORRIDOR_PADDING = HUMAN_BBOX_MARGIN  # 兼容原有变量，改为小margin

# 卡尔曼滤波参数
PROCESS_NOISE = 0.01
MEASURE_NOISE = 0.1

# 绘图颜色 BGR
COLOR_REAL = (255, 255, 255)    # 白色真实轨迹
COLOR_PRED = (0, 0, 255)        # 红色预测轨迹
COLOR_CORRIDOR = (255, 0, 0)    # 蓝色走廊多边形
ALPHA_CORRIDOR = 0.2

# ====================== 文件路径配置 ======================
INPUT_KEYPOINT_XLSX = "inputs/urfall_keypoints_adl-01-cam0.xlsx"
INPUT_SEQ_CSV = "inputs/sequence_split_recommendation.csv"
INPUT_VIDEO = "inputs/adl-01-cam0.mp4"

# 输出路径
OUT_DIR = "outputs/"
OUT_ALIGNED_CSV = OUT_DIR + "module1_aligned.csv"
OUT_FRAME_MAP_JSON = OUT_DIR + "module1_frame_map.json"
OUT_TRAJ_DATASET = OUT_DIR + "module2_traj_dataset.json"
OUT_KALMAN_PRED = OUT_DIR + "module3_predict_result.json"
OUT_ERROR_CSV = OUT_DIR + "pred_error_stat.csv"
OUT_STANDARD_JSON = OUT_DIR + "standard_pred_output.json"
OUT_VIS_VIDEO = OUT_DIR + "prediction_visual.mp4"