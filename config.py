import numpy as np

# ==========================================
# 1. 物理层与信道参数 (对标 3GPP TR 38.901)
# ==========================================
FC = 5.9                   # V2X 载波频率 (GHz)
FC_HZ = FC * 1e9           # Same carrier frequency in Hz (for utils.py compatibility)
BREAKPOINT_DISTANCE = 177.0  # 3GPP UMi LOS breakpoint distance (meters)
TX_POWER_DBM = 23.0        # 发射功率 (dBm) 
NOISE_FIGURE_DB = 9.0      # 噪声系数
BANDWIDTH_HZ = 10e6        # 10MHz 载波
THERMAL_NOISE_DENSITY = -174
NOISE_FLOOR_DBM = THERMAL_NOISE_DENSITY + 10 * np.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB

# ==========================================
# 2. 场景与移动性参数
# ==========================================
SIM_STEP_SECONDS = 0.1     # 每步 100ms
SPEED_MIN = 1.0            # 对应 Urban 拥堵
SPEED_MAX = 45.0           # 对应 Highway 高速 (162 km/h)

# ==========================================
# 3. TS 38.331 协议参数 (Stage 0 核心)
# ==========================================
HYSTERESIS_DB = 3.0        # 迟滞余量 (Hys)
TTT_STEPS = 3              # 触发观察窗口 (Time-to-Trigger)

# ==========================================
# 4. 强化学习 & AI模型参数
# ==========================================
# 💡 动作空间：标准化 5dB 间隔
ACTION_THRESHOLDS = [-115, -110, -105, -100, -95, -90, -85]
STATE_DIM = 5              # [RSRP, Delta_RSRP, Speed, Dist, ISD]
ACTION_DIM = len(ACTION_THRESHOLDS)

# --- 训练参数 ---
GAMMA = 0.99               
LEARNING_RATE = 1e-4       # 常用学习率，保证异构环境收敛稳定性
MEMORY_SIZE = 10000        
BATCH_SIZE = 256           # 每次采样批量大小  GPU可用后，可适当增大（128/256）              

# --- 探索策略 ---
EPSILON_START = 1.0   
EPSILON_END = 0.05         
EPSILON_DECAY = 0.997     

# --- 多种子统计保证 ---
MAX_EPISODES = 6000 # 训练总回合数       
# SEEDS = [71, 123, 456, 787, 45]  # 不同随机种子
SEEDS = [71, 123, 456]  # 不同随机种子（减少到2个，节约训练时间）