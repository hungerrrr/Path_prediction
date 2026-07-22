# 流式路径预测本地展示 Demo

该页面使用假设的摄像头姿态数据，展示以下完整流程：

- 流式关键点和人物框输入；
- 每人物 60 帧历史缓冲；
- GRU、方案 A、方案 B 切换；
- Top-1 与多候选路径、预测走廊；
- 标准输入/输出 JSON；
- 假设的下游跌倒风险融合结果。

## 推荐启动方式

在项目根目录执行：

```powershell
conda activate fall_detect
python demo\streaming_path_demo\serve.py
```

浏览器会自动打开 `http://127.0.0.1:8765/`。按 `Ctrl+C` 停止。

如果端口被占用：

```powershell
python demo\streaming_path_demo\serve.py --port 9000
```

如果不希望自动打开浏览器：

```powershell
python demo\streaming_path_demo\serve.py --no-browser
```

页面没有网络依赖，也可以直接双击 `index.html` 打开。

## 说明

当前页面用于产品展示和接口讨论，其中轨迹、延迟与跌倒风险为模拟值，
不会调用正式 PyTorch 模型。正式模型的真实流式接口位于
`src/streaming_predictor.py`。
