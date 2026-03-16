import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BaseAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def select_action(self, state, is_training=True):
        raise NotImplementedError # 由子类实现

    def store_transition(self, s, a, r, s_next, done):
        pass # 部分算法不需要 buffer

    def train_step(self):
        pass # 部分算法不需要训练步骤

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass