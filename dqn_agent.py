import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import config

# ==========================================
# 1. 神经网络结构 (Q-Network)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 简单的 3 层全连接网络
        # 输入: 状态向量 -> 输出: 每个动作的 Q 值
        # self.fc1 = nn.Linear(state_dim, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, action_dim)  #效果提升了%1.06 想提升更多，所以增加了神经元
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        # 使用 ReLU 激活函数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# 2. DQN 智能体 (Agent Logic)
# ==========================================
class DQNAgent:
    def __init__(self):
        self.state_dim = config.STATE_DIM
        self.action_dim = config.ACTION_DIM
        
        # 检查是否有 GPU，没有就用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化两个网络: 策略网络 (Policy) 和 目标网络 (Target)
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 同步初始参数
        self.target_net.eval() # 目标网络只预测，不训练
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        
        # 经验回放池 (Experience Replay Buffer)
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        # 探索参数 (Epsilon-Greedy)
        self.epsilon = config.EPSILON_START

    def select_action(self, state, is_training=True):
        """
        根据当前状态选择动作
        :param is_training: 如果是测试模式，完全利用 (Exploitation)
        """
        # 1. 探索 (Exploration): 随机瞎选
        if is_training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # 2. 利用 (Exploitation): 选 Q 值最大的
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """存入经验池"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        从经验池采样并训练一次网络
        """
        if len(self.memory) < config.BATCH_SIZE:
            return 0.0 # 样本不够，先不练
        
        # 1. 随机采样 Batch
        batch = random.sample(self.memory, config.BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        # 转为 Tensor
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # 2. 计算当前 Q 值 (Q_expected)
        # gather(1, action_batch) 意思是：只取我们在该步实际执行的那个动作对应的 Q 值
        q_expected = self.policy_net(state_batch).gather(1, action_batch)
        
        # 3. 计算目标 Q 值 (Q_target)
        # Q_target = r + gamma * max(Q_next)
        with torch.no_grad():
            q_next = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            q_target = reward_batch + (config.GAMMA * q_next * (1 - done_batch))
            
        # 4. 计算 Loss 并反向传播
        loss = self.loss_fn(q_expected, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 5. 更新 Epsilon (减少探索，增加利用)
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)
        
        return loss.item()

    def update_target_network(self):
        """定期同步目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))