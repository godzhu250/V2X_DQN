import numpy as np
import config
import utils  # 当前未直接使用，保留


class VehicleEnv:
    def __init__(self):
        self.action_space_n = config.ACTION_DIM
        self.state_dim = config.STATE_DIM
        self.last_rsrp = -80.0
        self.scenario_type = 'Highway'

        # 统计量
        self.handover_count = 0
        self.rlf_steps = 0
        self.pingpong_count = 0
        self.ttt_counter = 0
        self.time_step = 0
        self.last_ho_time = -999  # 用于判定乒乓切换的时间点
        self.steps_since_last_ho = 0

        self.current_rsrp = self.last_rsrp
        self.ue_relative_x = 0.0
        self.inter_relay_dist = 1200.0
        self.shadowing_std = 4.0
        self.ue_speed = 35.0

    def reset(self, force_scenario=None):
        self.handover_count = 0
        self.rlf_steps = 0
        self.pingpong_count = 0
        self.ttt_counter = 0
        self.time_step = 0
        self.last_ho_time = -999
        self.steps_since_last_ho = 0

        # 随机场景或按指定场景重置
        self.scenario_type = force_scenario if force_scenario else np.random.choice(['Highway', 'Urban'])

        if self.scenario_type == 'Highway':
            self.inter_relay_dist, self.shadowing_std, self.ue_speed = 1200.0, 4.0, 35.0
        else:
            self.inter_relay_dist, self.shadowing_std, self.ue_speed = 350.0, 8.0, 5.0

        self.ue_relative_x = np.random.uniform(0, 50)
        self.current_rsrp = self._calculate_physics_rsrp(self.ue_relative_x)
        self.last_rsrp = self.current_rsrp
        return self._get_state()

    def _calculate_physics_rsrp(self, distance):
        """严格遵循 3GPP TR 38.901 的路径损耗建模。"""
        d = max(1.0, distance)
        fc = config.FC
        if self.scenario_type == 'Urban':
            path_loss = 35.3 + 31.9 * np.log10(d) + 20 * np.log10(fc)
        else:
            path_loss = 32.4 + 21.0 * np.log10(d) + 20 * np.log10(fc)
        noise = np.random.normal(0, self.shadowing_std)
        return config.TX_POWER_DBM - path_loss + noise

    def step(self, action_index):
        selected_threshold = config.ACTION_THRESHOLDS[action_index]
        self.ue_relative_x += self.ue_speed * config.SIM_STEP_SECONDS

        rsrp_serving = self._calculate_physics_rsrp(self.ue_relative_x)
        dist_target = abs(self.inter_relay_dist - self.ue_relative_x)
        rsrp_target = self._calculate_physics_rsrp(dist_target)

        # 3GPP TTT 触发逻辑
        trigger_reselection = False
        if rsrp_serving < selected_threshold:
            self.ttt_counter += 1
        else:
            self.ttt_counter = 0

        if self.ttt_counter >= config.TTT_STEPS:
            if rsrp_target > (rsrp_serving + config.HYSTERESIS_DB):
                trigger_reselection = True

        ho_attempted = 1 if trigger_reselection else 0
        handover_occurred, handover_failed, is_pingpong = False, False, 0

        if trigger_reselection:
            if rsrp_target < -105:
                handover_failed = True
            else:
                # 乒乓判定：1秒内再次切换
                if self.time_step - self.last_ho_time < 10:
                    is_pingpong = 1
                    self.pingpong_count += 1

                self.last_ho_time = self.time_step
                self.ue_relative_x = max(5.0, abs(self.inter_relay_dist - self.ue_relative_x))
                rsrp_serving = self._calculate_physics_rsrp(self.ue_relative_x)

                # 切换成功计数
                self.handover_count += 1
                handover_occurred = True

            self.ttt_counter = 0

        if handover_occurred:
            self.steps_since_last_ho = 0
        else:
            self.steps_since_last_ho += 1

        # KPI 统计逻辑
        # HFR 只统计真实切换失败
        is_rlf = 1 if handover_failed else 0
        # 弱信号单独统计（不计入 HFR）
        is_weak_signal = 1 if rsrp_serving < -115 else 0
        # 仅用于环境内部链路失败惩罚
        is_link_fail = 1 if (is_weak_signal or handover_failed) else 0
        if is_link_fail:
            self.rlf_steps += 1

        # Reward 数值稳定化（不改变训练主流程与协议逻辑）
        reward = 0.0
        done = False

        # 1) 信号质量奖励：RSRP 映射到 [0,1] 再缩放
        if rsrp_serving > -115:
            q_norm = (rsrp_serving + 115.0) / 25.0
            q_norm = float(np.clip(q_norm, 0.0, 1.0))
            reward += 5.0 * q_norm

        # 2) 断连惩罚：触发后终止 episode
        if is_link_fail:
            reward -= 200.0
            done = True

        # 3) 切换惩罚：含递增项并封顶
        if handover_occurred:
            nth = max(0, self.handover_count - 1)
            ho_pen = 5.0 + 1.0 * min(nth, 10)
            reward -= ho_pen

        # 4) 乒乓额外惩罚
        if is_pingpong:
            reward -= 8.0

        # 5) 存活奖励
        if not done:
            reward += 0.5

        # 6) 长期不切换惩罚：仅在弱覆盖区（RSRP<-100 dBm）触发
        if (not done) and (rsrp_serving < -100) and (self.steps_since_last_ho > 80):
            stagnation_pen = 0.1 * (self.steps_since_last_ho - 80) / 100.0
            reward -= min(stagnation_pen, 1.0)

        # episode 终止条件
        self.current_rsrp = rsrp_serving
        self.time_step += 1
        if self.ue_relative_x > (self.inter_relay_dist + 200) or self.time_step > 500:
            done = True

        return self._get_state(), reward, done, {
            'rsrp': rsrp_serving,
            'is_rlf': is_rlf,
            'is_weak_signal': is_weak_signal,
            'is_pingpong': is_pingpong,
            'ho_attempted': ho_attempted,
            'ho_happened': 1 if handover_occurred else 0,
            'ttt': self.ttt_counter
        }

    def _get_state(self):
        delta = self.current_rsrp - self.last_rsrp
        self.last_rsrp = self.current_rsrp

        rsrp_norm = (self.current_rsrp + 115.0) / 25.0
        rsrp_norm = float(np.clip(rsrp_norm, 0.0, 1.0))

        return np.array([
            rsrp_norm,
            delta / 5.0,
            self.ue_speed / 45.0,
            self.ue_relative_x / 1200.0,
            self.inter_relay_dist / 1200.0
        ], dtype=np.float32)
