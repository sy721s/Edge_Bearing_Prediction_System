import torch
import torch.nn as nn
import numpy as np
import os
import signal
import sys
import json
import pandas as pd

# ------------------- 断点保存信号处理 -------------------
def receive_signal(signum, frame):
    global model, optimizer, epoch, loss, MODEL_TYPE
    print(f"\n[警告]收到终止信号，保存{MODEL_TYPE}断点")
    save_checkpoint(epoch, model, optimizer, loss, f'cwru_{MODEL_TYPE.lower()}_checkpoint.pth')
    sys.exit(0)

def save_checkpoint(epoch, model, optimizer, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)

signal.signal(signal.SIGTERM, receive_signal)
signal.signal(signal.SIGINT, receive_signal)

# ------------------- 模型定义 -------------------
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
        else:
            raise ValueError("仅支持RNN/GRU/LSTM")
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        bs = x.size(0)
        h0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(device)
        if self.cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, bs, self.hidden_size).to(device)
            out, (hn, cn) = self.rnn_core(x, (h0, c0))
        else:
            out, hn = self.rnn_core(x, h0)
        return self.fc(out)

# ------------------- CWRU轴承仿真振动数据加载 -------------------
def load_cwru_sim(device):
    t = np.linspace(0, 80, 6000)
    # 模拟正常振动+故障突变噪声时序
    raw = np.sin(0.2 * t) + 0.3 * np.cos(1.8 * t) + np.random.normal(0, 0.07, len(t))
    x = torch.tensor(raw[:-1], dtype=torch.float32).view(1, -1, 1).to(device)
    y = torch.tensor(raw[1:], dtype=torch.float32).view(1, -1, 1).to(device)
    return x, y

# ------------------- 批量训练入口 -------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = 200
    hidden_list = [4, 32, 128]
    model_list = ["RNN", "GRU", "LSTM"]
    all_loss_log = {}
    all_param_info = {}

    for hs in hidden_list:
        hs_key = f"hidden_{hs}"
        all_loss_log[hs_key] = {}
        all_param_info[hs_key] = {}
        for MODEL_TYPE in model_list:
            ckpt_path = f'cwru_{MODEL_TYPE.lower()}_hs{hs}_checkpoint.pth'
            weight_path = f'weights/cwru_{MODEL_TYPE.lower()}_hs{hs}_weights.pth'
            os.makedirs("weights", exist_ok=True)
            print(f"\n=====开始训练：{MODEL_TYPE}, HiddenSize={hs}=====")
            model = SleepRNNDemo(MODEL_TYPE, hidden_size=hs).to(device)
            # 统计模型总参数量（实验报告使用）
            total_params = sum(p.numel() for p in model.parameters())
            all_param_info[hs_key][MODEL_TYPE] = total_params
            print(f"{MODEL_TYPE} 参数总量：{total_params}")

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            start_epoch = 0
            loss = torch.tensor(0.)
            # 加载断点续训文件
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt['epoch'] + 1
                loss = ckpt['loss']

            x_data, y_data = load_cwru_sim(device)
            loss_rec = []
            try:
                for epoch in range(start_epoch, total_epochs):
                    model.train()
                    pred = model(x_data)
                    loss = criterion(pred, y_data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # 每10轮记录一次损失
                    if (epoch + 1) % 10 == 0:
                        loss_rec.append(round(loss.item(), 5))
                        print(f"Epoch{epoch+1}, Loss:{loss.item():.4f}")
                        save_checkpoint(epoch, model, optimizer, loss, ckpt_path)
                # 训练完成保存纯净权重，删除断点文件
                torch.save(model.state_dict(), weight_path)
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                all_loss_log[hs_key][MODEL_TYPE] = loss_rec
            except Exception as e:
                save_checkpoint(epoch, model, optimizer, loss, ckpt_path)
                print("训练异常退出，已保存断点", e)
    # 保存全部实验日志到exp_log文件夹
    os.makedirs("exp_log", exist_ok=True)
    with open("exp_log/loss_record.json", "w", encoding="utf-8") as f:
        json.dump(all_loss_log, f, indent=2)
    with open("exp_log/param_count.json", "w", encoding="utf-8") as f:
        json.dump(all_param_info, f, indent=2)
    print("全部模型训练完成，损失/参数日志已存入exp_log，权重文件存入weights")