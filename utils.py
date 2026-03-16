import numpy as np
import config

def calculate_pathloss_3gpp(distance_m):
    """
    3GPP TR 38.901 UMi LOS Pathloss Model
    修正了 Breakpoint Distance 逻辑，确保与参数一致。
    """
    dist = max(distance_m, 1.0)
    f_ghz = config.FC_HZ / 1e9
    
    # 阴影衰落
    shadowing_std = 6.0 
    shadowing = np.random.normal(0, shadowing_std)

    # 动态读取配置中的断点距离 (约 177m)
    d_bp = config.BREAKPOINT_DISTANCE
    
    if dist <= d_bp:
        # PL1: Close range
        pl = 32.4 + 21 * np.log10(dist) + 20 * np.log10(f_ghz)
    else:
        # PL2: Far range (衰减更快)
        pl_at_bp = 32.4 + 21 * np.log10(d_bp) + 20 * np.log10(f_ghz)
        pl = pl_at_bp + 40 * np.log10(dist / d_bp)
        
    return pl + shadowing

def calculate_rsrp(distance_m, tx_power_dbm=config.TX_POWER_DBM):
    path_loss = calculate_pathloss_3gpp(distance_m)
    rsrp = tx_power_dbm - path_loss
    return rsrp