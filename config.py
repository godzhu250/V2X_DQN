import numpy as np

# ============================================================
# Scenario configuration
# ============================================================
# Valid values: "Highway" or "Urban"
SCENARIO = "Highway"

SIM_STEP_SECONDS = 0.1
EPISODE_STEPS = 500

SCENARIO_PROFILES = {
    "Highway": {
        "speed_mps": 120.0 / 3.6,  # 33.333... m/s
        "inter_relay_dist_m": 1400.0,
        "shadowing_std_db": 6.0,
    },
    "Urban": {
        "speed_mps": 60.0 / 3.6,  # 16.666... m/s
        "inter_relay_dist_m": 350.0,
        "shadowing_std_db": 8.0,
    },
}

SPEED_HIGHWAY_MPS = SCENARIO_PROFILES["Highway"]["speed_mps"]
SPEED_URBAN_MPS = SCENARIO_PROFILES["Urban"]["speed_mps"]

# ============================================================
# Radio / channel parameters
# ============================================================
FC = 5.9
FC_HZ = FC * 1e9
BREAKPOINT_DISTANCE = 177.0
TX_POWER_DBM = 23.0
NOISE_FIGURE_DB = 9.0
BANDWIDTH_HZ = 10e6
THERMAL_NOISE_DENSITY = -174
NOISE_FLOOR_DBM = THERMAL_NOISE_DENSITY + 10 * np.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB
RSRP_NOISE_STD_DB = 3.0

# ============================================================
# 3GPP reselection core logic
# ============================================================
HYSTERESIS_DB = 3.0
TTT_SECONDS = 0.3
TTT_STEPS = int(round(TTT_SECONDS / SIM_STEP_SECONDS))
TARGET_RSRP_FAIL_THRESHOLD_DBM = -80.0
SUCCESS_MARGIN_DB = 1.0
PINGPONG_WINDOW_STEPS = 200  # 20 seconds with 0.1s step
WEAK_SIGNAL_THRESHOLD_DBM = -105.0
HO_VALIDATION_STEPS = 5
VALIDATION_MIN_SUCCESS_RATIO = 0.6
POST_HO_POSITION_OFFSET_M = -70.0

# ============================================================
# RL action/state/training
# ============================================================
ACTION_THRESHOLDS = [-115, -110, -105, -100, -95, -90, -85]
STATE_DIM = 5  # [RSRP_norm, Delta_RSRP_norm, Speed_norm, Position_norm, ISD_norm]
ACTION_DIM = len(ACTION_THRESHOLDS)

GAMMA = 0.99
LEARNING_RATE = 1e-4
MEMORY_SIZE = 10000
BATCH_SIZE = 256

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.997
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 350000 # 会在350000/500=700个episode内探索率从1.0衰减到0.05

TARGET_UPDATE_INTERVAL = 10
TRAIN_FREQ = 4

# ============================================================
# Reward weights (explicitly configurable)
# r = reward_qos*qnorm
#     - reward_fail*ho_failed
#     - (reward_ho_cost_base + reward_ho_cost_scale*nHO)*ho_success
#     - reward_pingpong*pingpong
#     - reward_weak_signal*weak_signal_event
#     + reward_alive
#     - reward_stagnation*stagnation_term
# ============================================================
REWARD_QOS = 0.5
REWARD_FAIL = 200.0
REWARD_HO_COST_BASE = 5.0
REWARD_HO_COST_SCALE = 1.0
REWARD_HO_COST_N_CAP = 10
REWARD_PINGPONG = 8.0
REWARD_WEAK_SIGNAL = 2.0
REWARD_ALIVE = 0.5
REWARD_STAGNATION = 1.0
STAGNATION_START_STEPS = 80
STAGNATION_RSRP_THRESHOLD_DBM = -100.0
STAGNATION_NORMALIZER = 100.0
STAGNATION_PENALTY_BASE = 0.2
STAGNATION_DECAY_EPISODES = 600

# ============================================================
# Experiment defaults
# ============================================================
HIGHWAY_SEEDS = [71, 123, 456]
HIGHWAY_EPISODES = 3000

URBAN_SEEDS = [71, 123, 456]
URBAN_EPISODES = 3000

BASELINE_EVAL_EPISODES = 200
BASELINE_SEED = 250
BASELINE_MIN_TOTAL_ATTEMPTS = 20
EVAL_INTERVAL_EPISODES = 100
EVAL_EPISODES = 100
