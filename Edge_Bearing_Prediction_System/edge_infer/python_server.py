import torch
import torch.nn as nn
import numpy as np
import asyncio
import json
import time
import random
import os
import pandas as pd
import websockets

# 推理配置区，可自行修改
MODEL_TYPE = "LSTM"
HIDDEN_SIZE = 32
WEIGHTS_PATH = f'weights/cwru_{MODEL_TYPE.lower()}_hs{HIDDEN_SIZE}_weights.pth'
HOST = "127.0.0.1"
PORT = 8765

# 和训练端完全一致的模型结构
class SleepRNNDemo(nn.Module):
    def __init__(self, cell_type='LSTM', input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if self.cell_type == 'RNN':
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'GRU':
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'LSTM':
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        bs = x.size(0)
        h0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(device)
        if self.cell_type == "LSTM":
            c0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(device)
            out, (hn, cn) = self.rnn_core(x, (h0, c0))
        else:
            out, hn = self.rnn_core(x, h0)
        return self.fc(out)

latency_list = []

async def stream_data(websocket):
    print(f"客户端已连接，加载模型 {MODEL_TYPE} HiddenSize={HIDDEN_SIZE}")
    model = SleepRNNDemo(MODEL_TYPE, hidden_size=HIDDEN_SIZE)
    # 加载训练好的权重
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("权重加载成功")
    except Exception as e:
        print(f"权重文件缺失，使用随机初始化模型，报错信息：{e}")

    # 仿真轴承振动测试时序，和训练数据分布一致
    t = np.linspace(0, 120, 3500)
    test_sig = np.sin(0.2 * t) + 0.3 * np.cos(1.8 * t) + np.random.normal(0, 0.07, len(t))

    with torch.no_grad():
        for i in range(len(test_sig) - 1):
            start_time = time.time()
            # 构造单步输入
            inp = torch.tensor([[[test_sig[i]]]], dtype=torch.float32)
            pred = model(inp).item()
            real_val = test_sig[i+1]
            # 计算推理耗时
            pure_cost_ms = (time.time() - start_time) * 1000
            # 模拟工业网络延迟波动
            total_latency = pure_cost_ms + random.uniform(5, 15)
            latency_list.append(total_latency)
            abs_error = abs(real_val - pred)

            # 向前端发送实时数据
            send_data = {
                "timestamp": time.time() * 1000,
                "model_type": MODEL_TYPE,
                "ch1_actual": float(real_val),
                "ch2_predict": float(pred),
                "error_abs": float(abs_error),
                "latency_ms": round(total_latency, 2),
                "hidden": HIDDEN_SIZE
            }
            await websocket.send(json.dumps(send_data))
            await asyncio.sleep(0.03)

    # 推理结束，保存全部时延数据到csv
    os.makedirs("exp_log", exist_ok=True)
    save_file = f"exp_log/{MODEL_TYPE}_hs{HIDDEN_SIZE}_latency.csv"
    pd.DataFrame({"latency_ms": latency_list}).to_csv(save_file, index=False)
    print(f"推理时延统计文件已保存至：{save_file}")
    print(f"本次平均推理时延：{np.mean(latency_list):.2f} ms")

async def main():
    async with websockets.serve(stream_data, HOST, PORT):
        print(f"边缘WebSocket推理服务启动完成，地址：ws://{HOST}:{PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())