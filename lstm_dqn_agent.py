import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import config
from base_agent import BaseAgent

# ==========================================
# 1. 递归神经网络结构 (LSTM + Linear)
# ==========================================
class LSTMNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(LSTMNetwork, self).__init__()
        self.hidden_dim = 128
        
        # 特征映射层
        self.fc1 = nn.Linear(state_dim, 128)
        
        # LSTM 层：处理序列信息
        # batch_first=True 表示输入维度为 (batch, seq_len, features)
        self.lstm = nn.LSTM(128, self.hidden_dim, batch_first=True)
        
        # 输出层
        self.fc2 = nn.Linear(self.hidden_dim, action_dim)

    def forward(self, x):
        # x 维度: (Batch, Seq_Len, State_Dim)
        x = torch.relu(self.fc1(x))
        
        # LSTM 返回 (output, (h_n, c_n))，我们只取序列最后一个时间点的输出
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :] # 取序列最后一帧
        
        q_values = self.fc2(last_time_step_out)
        return q_values

# ==========================================
# 2. DRQN 智能体
# ==========================================
class LSTMDQNAgent(BaseAgent):
    def __init__(self):
        super(LSTMDQNAgent, self).__init__()
        self.state_dim = config.STATE_DIM
        self.action_dim = config.ACTION_DIM
        self.seq_len = 8 # 💡 记忆长度：看过去 8 帧的状态
        
        self.policy_net = LSTMNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = LSTMNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        
        # 经验池：为了保持接口统一，我们在 Agent 内部处理序列逻辑
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.epsilon = config.EPSILON_START
        
        # 用于在测试/运行期间暂存当前回合的历史
        self.state_sequence = deque(maxlen=self.seq_len)

    def _get_padded_seq(self, seq):
        """如果序列不够长，前面补零"""
        curr_seq = list(seq)
        while len(curr_seq) < self.seq_len:
            curr_seq.insert(0, np.zeros(self.state_dim))
        return np.array(curr_seq)

    def select_action(self, state, is_training=True):
        # 将当前状态加入历史
        self.state_sequence.append(state)
        
        if is_training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            # 准备序列数据 (1, Seq_Len, State_Dim)
            seq_data = self._get_padded_seq(self.state_sequence)
            seq_tensor = torch.FloatTensor(seq_data).unsqueeze(0).to(self.device)
            q_values = self.policy_net(seq_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        💡 技巧：为了不改 train.py 的循环，我们在存储时记录序列
        """
        # 注意：state 已在 select_action 中追加，这里不再重复追加，避免时序重复
        # 在学术论文中，DRQN 推荐存储 Episode 或 Trajectory
        # state is already appended in select_action; do not append it again here.
        # Keeping sequence order clean avoids duplicated timesteps for LSTM training.
        curr_seq = self._get_padded_seq(self.state_sequence)
        
        temp_seq = list(self.state_sequence)
        temp_seq.append(next_state)
        next_seq = self._get_padded_seq(temp_seq[1:])
        
        self.memory.append((curr_seq, action, reward, next_seq, done))

    def train_step(self):
        if len(self.memory) < config.BATCH_SIZE:
            return 0.0
        
        batch = random.sample(self.memory, config.BATCH_SIZE)
        # s_batch 现在的形状是 (Batch, Seq_Len, State_Dim)
        s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)
        
        s_batch = torch.FloatTensor(np.array(s_batch)).to(self.device)
        a_batch = torch.LongTensor(a_batch).unsqueeze(1).to(self.device)
        r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(self.device)
        ns_batch = torch.FloatTensor(np.array(ns_batch)).to(self.device)
        d_batch = torch.FloatTensor(d_batch).unsqueeze(1).to(self.device)
        
        q_expected = self.policy_net(s_batch).gather(1, a_batch)
        
        with torch.no_grad():
            q_next = self.target_net(ns_batch).max(1)[0].unsqueeze(1)
            q_target = r_batch + (config.GAMMA * q_next * (1 - d_batch))
            
        loss = self.loss_fn(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
