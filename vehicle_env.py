import numpy as np

import config


class VehicleEnv:
    """Single-scenario NR-V2X reselection environment.

    Episode length is fixed to config.EPISODE_STEPS.
    """

    def __init__(self, scenario=None):
        self.action_space_n = config.ACTION_DIM
        self.state_dim = config.STATE_DIM

        self.scenario_type = self._validate_scenario(scenario or config.SCENARIO)
        self._apply_scenario_profile()

        self.max_inter_relay_dist = max(
            p["inter_relay_dist_m"] for p in config.SCENARIO_PROFILES.values()
        )
        self.max_speed = max(p["speed_mps"] for p in config.SCENARIO_PROFILES.values())

        self.last_rsrp = -80.0
        self.current_rsrp = self.last_rsrp
        self.ue_relative_x = 0.0

        self.ttt_counter = 0
        self.time_step = 0
        self.last_success_ho_time = -10**9
        self.steps_since_last_ho = 0
        self.current_relay_id = 0
        self.prev_relay_id_before_last_success_ho = None

        self.episode_stats = {}

    def _validate_scenario(self, scenario):
        if scenario not in config.SCENARIO_PROFILES:
            raise ValueError(
                f"Invalid scenario '{scenario}'. "
                f"Use one of {list(config.SCENARIO_PROFILES.keys())}."
            )
        return scenario

    def _apply_scenario_profile(self):
        profile = config.SCENARIO_PROFILES[self.scenario_type]
        self.inter_relay_dist = float(profile["inter_relay_dist_m"])
        self.shadowing_std = float(profile["shadowing_std_db"])
        self.ue_speed = float(profile["speed_mps"])

    def reset(self, force_scenario=None):
        if force_scenario is not None:
            self.scenario_type = self._validate_scenario(force_scenario)
            self._apply_scenario_profile()

        self.ttt_counter = 0
        self.time_step = 0
        self.last_success_ho_time = -10**9
        self.steps_since_last_ho = 0
        self.current_relay_id = 0
        self.prev_relay_id_before_last_success_ho = None

        self.episode_stats = {
            "ho_attempted": 0,
            "ho_success": 0,
            "ho_failed": 0,
            "pingpong": 0,
            "weak_signal_event": 0,
        }

        self.ue_relative_x = np.random.uniform(0.0, 50.0)
        self.current_rsrp = self._calculate_physics_rsrp(self.ue_relative_x)
        self.last_rsrp = self.current_rsrp

        return self._get_state()

    def _calculate_physics_rsrp(self, distance):
        d = max(1.0, distance)
        fc = config.FC

        if self.scenario_type == "Urban":
            path_loss = 35.3 + 31.9 * np.log10(d) + 20.0 * np.log10(fc)
        else:
            path_loss = 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(fc)

        noise = np.random.normal(0.0, self.shadowing_std)
        return config.TX_POWER_DBM - path_loss + noise

    def _episode_kpi(self):
        attempts = max(self.episode_stats["ho_attempted"], 1)
        hfr = self.episode_stats["ho_failed"] / attempts
        ppr = self.episode_stats["pingpong"] / attempts
        ehr = 1.0 - hfr - ppr
        return {
            "hfr": float(hfr),
            "ppr": float(ppr),
            "ehr": float(ehr),
        }

    def get_episode_stats(self):
        stats = dict(self.episode_stats)
        stats.update(self._episode_kpi())
        return stats

    def step(self, action_index):
        if action_index < 0 or action_index >= self.action_space_n:
            raise ValueError(f"Invalid action_index: {action_index}")

        selected_threshold = config.ACTION_THRESHOLDS[action_index]
        self.ue_relative_x += self.ue_speed * config.SIM_STEP_SECONDS

        # This environment keeps the original dual-relay geometric flip backbone:
        # ue_relative_x is interpreted as distance to current serving relay.
        # Target relay is the opposite relay (1 - current_relay_id).
        serving_relay_id = self.current_relay_id
        target_relay_id = 1 - serving_relay_id

        dist_serving = self.ue_relative_x
        dist_target = abs(self.inter_relay_dist - self.ue_relative_x)
        rsrp_serving = self._calculate_physics_rsrp(dist_serving)
        rsrp_target = self._calculate_physics_rsrp(dist_target)

        trigger_reselection = False
        if rsrp_serving < selected_threshold:
            self.ttt_counter += 1
        else:
            self.ttt_counter = 0

        # Attempt definition: threshold + TTT only.
        if self.ttt_counter >= config.TTT_STEPS:
            trigger_reselection = True

        ho_attempted = 1 if trigger_reselection else 0
        ho_success = 0
        ho_failed = 0
        pingpong = 0

        if trigger_reselection:
            target_quality_ok = rsrp_target >= config.TARGET_RSRP_FAIL_THRESHOLD_DBM
            target_has_margin = rsrp_target >= (rsrp_serving + config.SUCCESS_MARGIN_DB)

            if target_quality_ok and target_has_margin:
                ho_success = 1

                old_relay_id = self.current_relay_id
                new_relay_id = 1 - old_relay_id

                # Ping-pong: successful HO that returns to previous relay in short window.
                if (
                    self.prev_relay_id_before_last_success_ho is not None
                    and new_relay_id == self.prev_relay_id_before_last_success_ho
                    and (self.time_step - self.last_success_ho_time) < config.PINGPONG_WINDOW_STEPS
                ):
                    pingpong = 1

                self.prev_relay_id_before_last_success_ho = old_relay_id
                self.current_relay_id = new_relay_id
                self.last_success_ho_time = self.time_step

                # Keep original geometry flip and synchronize with relay-id update.
                self.ue_relative_x = max(5.0, abs(self.inter_relay_dist - self.ue_relative_x))
                rsrp_serving = self._calculate_physics_rsrp(self.ue_relative_x)
            else:
                ho_failed = 1

            self.ttt_counter = 0

        if ho_success:
            self.steps_since_last_ho = 0
        else:
            self.steps_since_last_ho += 1

        weak_signal_event = 1 if rsrp_serving < config.WEAK_SIGNAL_THRESHOLD_DBM else 0

        self.episode_stats["ho_attempted"] += ho_attempted
        self.episode_stats["ho_success"] += ho_success
        self.episode_stats["ho_failed"] += ho_failed
        self.episode_stats["pingpong"] += pingpong
        self.episode_stats["weak_signal_event"] += weak_signal_event

        q_norm = np.clip(
            (rsrp_serving - config.WEAK_SIGNAL_THRESHOLD_DBM) / 25.0,
            0.0,
            1.0,
        )

        reward = config.REWARD_QOS * float(q_norm)
        reward -= config.REWARD_FAIL * ho_failed

        if ho_success:
            n_ho_done = self.episode_stats["ho_success"]
            ho_penalty = config.REWARD_HO_COST_BASE + config.REWARD_HO_COST_SCALE * min(
                max(0, n_ho_done - 1),
                config.REWARD_HO_COST_N_CAP,
            )
            reward -= ho_penalty

        reward -= config.REWARD_PINGPONG * pingpong
        reward -= config.REWARD_WEAK_SIGNAL * weak_signal_event
        reward += config.REWARD_ALIVE

        if (
            rsrp_serving < config.STAGNATION_RSRP_THRESHOLD_DBM
            and self.steps_since_last_ho > config.STAGNATION_START_STEPS
        ):
            stagnation_term = (self.steps_since_last_ho - config.STAGNATION_START_STEPS) / max(
                1.0,
                config.STAGNATION_NORMALIZER,
            )
            reward -= config.REWARD_STAGNATION * min(stagnation_term, 1.0)

        self.current_rsrp = rsrp_serving
        self.time_step += 1

        done = self.time_step >= config.EPISODE_STEPS

        info = {
            "scenario": self.scenario_type,
            "step": self.time_step,
            "threshold_dbm": selected_threshold,
            "rsrp": float(rsrp_serving),
            "target_rsrp": float(rsrp_target),
            "serving_relay_id": int(serving_relay_id),
            "target_relay_id": int(target_relay_id),
            "ho_attempted": ho_attempted,
            "ho_success": ho_success,
            "ho_failed": ho_failed,
            "pingpong": pingpong,
            "weak_signal_event": weak_signal_event,
            "episode_ho_attempted": self.episode_stats["ho_attempted"],
            "episode_ho_success": self.episode_stats["ho_success"],
            "episode_ho_failed": self.episode_stats["ho_failed"],
            "episode_pingpong": self.episode_stats["pingpong"],
            "episode_weak_signal_event": self.episode_stats["weak_signal_event"],
        }
        info.update(self._episode_kpi())

        return self._get_state(), float(reward), done, info

    def _get_state(self):
        delta = self.current_rsrp - self.last_rsrp
        self.last_rsrp = self.current_rsrp

        rsrp_norm = np.clip(
            (self.current_rsrp - config.WEAK_SIGNAL_THRESHOLD_DBM) / 25.0,
            0.0,
            1.0,
        )

        return np.array(
            [
                rsrp_norm,
                delta / 5.0,
                self.ue_speed / max(self.max_speed, 1e-6),
                self.ue_relative_x / max(self.max_inter_relay_dist, 1e-6),
                self.inter_relay_dist / max(self.max_inter_relay_dist, 1e-6),
            ],
            dtype=np.float32,
        )
