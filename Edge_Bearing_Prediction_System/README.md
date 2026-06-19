# 基于边缘计算的工业轴承时序预测与异常预警系统
深度学习期末大作业 | 云端训练 + 边缘WebSocket推理 + D3工业监控面板全栈项目
选题：工业物联网与预测性维护（凯斯西储CWRU轴承故障时序分析）

## 一、项目简介
本项目使用 RNN、GRU、LSTM 三类循环神经网络完成工业轴承振动时序预测与故障实时预警，完成课程三大考核目标：
1. 模型选型对比：复现RNN/GRU/LSTM，对比梯度消失、收敛曲线、参数量；
2. 边缘场景适配：多组HiddenSize消融实验，测试推理时延、精度、资源占用；
3. 系统落地实证：改造D3前端工业示波器，实现毫秒级故障告警，完成工程验证。

项目完整流程：云端批量训练模型 → 边缘设备轻量化流式推理 → 前端可视化实时监测。

## 二、环境依赖
Python >= 3.8
一键安装依赖：
```bash
pip install -r requirements.txt

三、项目目录结构
Edge_Bearing_Prediction_System/
├─ README.md                # 项目说明文档
├─ requirements.txt         # 依赖配置
├─ .gitignore               # 忽略缓存文件
├─ train/
│  └─ SleepRNNDemo.py       # 云端批量训练代码
├─ edge_infer/
│  └─ python_server.py      # 边缘WebSocket推理服务
├─ frontend/
│  └─ index.html            # D3.js工业监控面板
├─ weights/                 # 训练自动生成模型权重.pth
├─ exp_log/                 # 实验日志、损失、时延csv
└─ docs/
   ├─ 期末大作业论文.docx   # 课程完整论文
   └─ 实验截图合集/         # 波形对比、运行截图

四、运行步骤
1. 云端批量训练模型
bash
运行
cd train
python SleepRNNDemo.py
自动遍历模型 (RNN/GRU/LSTM) 与隐藏层 (4/32/128) 训练，支持 Ctrl+C 断点保存。
2. 启动边缘推理服务
修改 python_server.py 中 MODEL_TYPE、HIDDEN_SIZE 参数后运行：
bash
运行
cd edge_infer
python python_server.py
WebSocket 地址：ws://127.0.0.1:8765
3. 前端可视化
浏览器直接打开 frontend/index.html
青色实线：真实轴承振动信号
紫色虚线：AI 预测波形
误差＞0.3 面板变红，触发故障预警
实时展示模型、隐藏层、推理延迟、预测误差
五、核心实验结论
梯度消失：RNN 预测波形滞后、振幅衰减；LSTM 拟合最优；GRU 性能接近 LSTM。
参数对比 (HiddenSize=32)：GRU 相比 LSTM 参数量减少约 28%，精度损失极小。
边缘适配：
HiddenSize=4：欠拟合，预警失效；
HiddenSize=32：平均时延＜20ms，满足工业 0.1s 报警要求；
HiddenSize=128：精度最高，但边缘设备内存占用高、延迟超标。
最优工程方案：GRU + HiddenSize=32，为本项目性价比最优方案。
六、课程考核对应实验
RNN/GRU/LSTM 对比，梯度消失、收敛曲线可视化；
多 HiddenSize 消融实验，权衡精度与时延；
前端工业示波器改造，完整系统落地验证；
Checkpoint 断点文件与纯净权重文件对比；
隐状态清零对照实验，验证时序记忆作用。
七、作者信息
姓名：莫诗颖
学号：32302440
专业班级：人工智能2301