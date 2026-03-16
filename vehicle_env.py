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

        self._reset_pending_state()
        self.episode_stats = {}

    def _reset_pending_state(self):
        self.pending_ho = False
        self.pending_target_relay_id = None
        self.pending_window_remaining = 0
        self.pending_success_counter = 0
        self.pending_failure_counter = 0
        self.pending_old_relay_id = None

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
        self._reset_pending_state()

        self.episode_stats = {
            "ho_attempted": 0,
            "ho_success": 0,
            "ho_failed": 0,
            "pingpong": 0,
            "weak_signal_event": 0,
            "pending_validation_started": 0,
            "validation_success": 0,
            "validation_failure": 0,
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

    def _distance_to_relay(self, relay_id):
        # Minimal-change geometry: keep mirrored backbone and bind serving/target via relay id.
        if relay_id == self.current_relay_id:
            return self.ue_relative_x
        return abs(self.inter_relay_dist - self.ue_relative_x)

    def _compute_serving_target_rsrp(self, serving_relay_id, target_relay_id):
        dist_serving = self._distance_to_relay(serving_relay_id)
        dist_target = self._distance_to_relay(target_relay_id)
        rsrp_serving = self._calculate_physics_rsrp(dist_serving)
        rsrp_target = self._calculate_physics_rsrp(dist_target)
        return rsrp_serving, rsrp_target

    def _target_condition(self, rsrp_serving, rsrp_target):
        target_quality_ok = rsrp_target >= config.TARGET_RSRP_FAIL_THRESHOLD_DBM
        target_has_margin = rsrp_target >= (rsrp_serving + config.SUCCESS_MARGIN_DB)
        return bool(target_quality_ok and target_has_margin)

    def _episode_kpi(self):
        attempts = int(self.episode_stats["ho_attempted"])
        if attempts == 0:
            hfr = np.nan
            ppr = np.nan
            ehr = np.nan
            no_attempt_episode = 1
        else:
            hfr = self.episode_stats["ho_failed"] / attempts
            ppr = self.episode_stats["pingpong"] / attempts
            ehr = 1.0 - hfr - ppr
            no_attempt_episode = 0
        return {
            "hfr": float(hfr),
            "ppr": float(ppr),
            "ehr": float(ehr),
            "no_attempt_episode": int(no_attempt_episode),
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

        serving_relay_id = self.current_relay_id
        if self.pending_ho and self.pending_target_relay_id is not None:
            target_relay_id = self.pending_target_relay_id
        else:
            target_relay_id = 1 - serving_relay_id

        rsrp_serving, rsrp_target = self._compute_serving_target_rsrp(
            serving_relay_id,
            target_relay_id,
        )

        ho_attempted = 0
        ho_success = 0
        ho_failed = 0
        pingpong = 0

        if self.pending_ho:
            if self._target_condition(rsrp_serving, rsrp_target):
                self.pending_success_counter += 1
            else:
                self.pending_failure_counter += 1

            self.pending_window_remaining -= 1

            if self.pending_window_remaining <= 0:
                success_ratio = self.pending_success_counter / max(1, int(config.HO_VALIDATION_STEPS))

                if success_ratio >= float(config.VALIDATION_MIN_SUCCESS_RATIO):
                    ho_success = 1

                    old_relay_id = (
                        self.pending_old_relay_id
                        if self.pending_old_relay_id is not None
                        else self.current_relay_id
                    )
                    new_relay_id = (
                        self.pending_target_relay_id
                        if self.pending_target_relay_id is not None
                        else (1 - old_relay_id)
                    )

                    if (
                        self.prev_relay_id_before_last_success_ho is not None
                        and new_relay_id == self.prev_relay_id_before_last_success_ho
                        and (self.time_step - self.last_success_ho_time) < config.PINGPONG_WINDOW_STEPS
                    ):
                        pingpong = 1

                    self.prev_relay_id_before_last_success_ho = old_relay_id
                    self.current_relay_id = new_relay_id
                    self.last_success_ho_time = self.time_step

                    mirrored_x = abs(self.inter_relay_dist - self.ue_relative_x)
                    # Coordinate-equivalent handling:
                    # in this local-distance representation, larger x means less stable serving.
                    # therefore a negative config.POST_HO_POSITION_OFFSET_M should push x upward.
                    unstable_shift = -config.POST_HO_POSITION_OFFSET_M
                    self.ue_relative_x = max(
                        5.0,
                        mirrored_x + unstable_shift,
                    )

                    rsrp_serving, rsrp_target = self._compute_serving_target_rsrp(
                        self.current_relay_id,
                        1 - self.current_relay_id,
                    )
                    self.episode_stats["validation_success"] += 1
                else:
                    ho_failed = 1
                    self.episode_stats["validation_failure"] += 1

                self._reset_pending_state()
        else:
            if rsrp_serving < selected_threshold:
                self.ttt_counter += 1
            else:
                self.ttt_counter = 0

            if self.ttt_counter >= config.TTT_STEPS:
                ho_attempted = 1
                self.pending_ho = True
                self.pending_target_relay_id = 1 - self.current_relay_id
                self.pending_window_remaining = int(config.HO_VALIDATION_STEPS)
                self.pending_success_counter = 0
                self.pending_failure_counter = 0
                self.pending_old_relay_id = self.current_relay_id
                self.episode_stats["pending_validation_started"] += 1
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
            "pending_ho": int(self.pending_ho),
            "pending_window_remaining": int(self.pending_window_remaining),
            "pending_success_counter": int(self.pending_success_counter),
            "pending_failure_counter": int(self.pending_failure_counter),
            "episode_ho_attempted": self.episode_stats["ho_attempted"],
            "episode_ho_success": self.episode_stats["ho_success"],
            "episode_ho_failed": self.episode_stats["ho_failed"],
            "episode_pingpong": self.episode_stats["pingpong"],
            "episode_weak_signal_event": self.episode_stats["weak_signal_event"],
            "episode_pending_validation_started": self.episode_stats["pending_validation_started"],
            "episode_validation_success": self.episode_stats["validation_success"],
            "episode_validation_failure": self.episode_stats["validation_failure"],
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
