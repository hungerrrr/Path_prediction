"""批量路径预测入口的兼容别名。

新集成应按使用场景调用 ``run_path_prediction_frame.py``（单帧请求）或
``run_path_prediction_batch.py``（完整骨架文件）。
"""

from run_path_prediction_batch import main


if __name__ == "__main__":
    main()
