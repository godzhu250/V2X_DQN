# V2X_DQN (Single-Scenario Pipeline)

This project keeps a single-scenario workflow for SL-RSRP threshold adaptation:
- Highway-only
- Urban-only

## Core Structure

- `config.py`
- `vehicle_env.py`
- `base_agent.py`
- `dqn_agent.py`
- `lstm_dqn_agent.py`
- `01_baseline_highway.py`
- `01_baseline_urban.py`
- `02_main.py` (shared training core)
- `02_train_highway.py` (Highway entry)
- `02_train_urban.py` (Urban entry)
- `03_plot_paper_figures.py`

## Episode Length

- `EPISODE_STEPS = 500`
- `SIM_STEP_SECONDS = 0.1`
- each episode = `500 * 0.1 = 50 seconds`

Episode termination remains fixed-length only.

## KPI Semantics

Per-episode KPI:
- if `ho_attempted == 0`: `HFR/PPR/EHR = NaN`, `no_attempt_episode = 1`
- else:
  - `HFR = ho_failed / ho_attempted`
  - `PPR = pingpong / ho_attempted`
  - `EHR = 1 - HFR - PPR`

So no-attempt episodes are excluded from per-episode KPI interpretation.
No-attempt episodes keep NaN semantics and are not remapped to 0.5/1.0.

Aggregate KPI (used for final bars):
- `aggregate_HFR = sum(ho_failed) / sum(ho_attempted)`
- `aggregate_PPR = sum(pingpong) / sum(ho_attempted)`
- `aggregate_EHR = 1 - aggregate_HFR - aggregate_PPR`

## Attempt / Validation / Success Logic

Attempt definition:
- serving RSRP below selected threshold
- condition holds for TTT duration

After attempt trigger, HO is **not** settled instantly.
The environment enters a validation window:
- `HO_VALIDATION_STEPS = 5`
- `VALIDATION_MIN_SUCCESS_RATIO = 0.6`

Validation target condition per step:
- target quality above `TARGET_RSRP_FAIL_THRESHOLD_DBM`
- target superiority at least `SUCCESS_MARGIN_DB`

HO result:
- success if validation success ratio >= `VALIDATION_MIN_SUCCESS_RATIO`
- otherwise failure

This keeps attempt and success/failure decoupled, while avoiding instant-success behavior.

## Ping-Pong Definition

Ping-pong is counted only when all are true:
- current HO is a **successful** HO
- there exists a previous successful HO
- current success returns to the relay used before last successful HO
- time gap from last successful HO < `PINGPONG_WINDOW_STEPS`

## Geometry and Relay Consistency

The environment still uses simplified two-relay mirrored geometry.
It keeps the original mirrored backbone (minimal-change design), but with relay-id-consistent serving/target logic.

To avoid overly idealized instant stabilization after success:
- mirrored position is kept
- plus post-HO offset:
  - `POST_HO_POSITION_OFFSET_M = -60.0`
  - implemented with coordinate-equivalent direction (in current local-distance coordinates,
    negative offset is mapped to a shift that keeps post-HO state closer to boundary, not deeper in comfort zone)

This is a minimal correction to make validation-based dynamics more realistic.

## Trend Source

Trend figures are generated from checkpoint evaluation trends, not raw training episodes:
- per-seed: `seed_<seed>_eval_trend.csv`
- model-level plotting source: `model_eval_trend.csv`

This keeps paper-style trend curves stable while preserving KPI semantics.

## Sensitivity and Calibration

- `TTT_SECONDS = 0.3`
- `TTT_STEPS = int(round(TTT_SECONDS / SIM_STEP_SECONDS))`
- `BASELINE_MIN_TOTAL_ATTEMPTS = 20`
- `PINGPONG_WINDOW_STEPS = 150`
- `WEAK_SIGNAL_THRESHOLD_DBM = -105.0` (calibrated to avoid all-zero weak-signal stats)
- `TARGET_RSRP_FAIL_THRESHOLD_DBM = -80.0` (stricter validation to avoid instant-success bias)
- Highway difficulty is mildly increased for non-trivial policies:
  - `inter_relay_dist_m = 1400.0`
  - `shadowing_std_db = 6.0`
- UE initial position is randomized over a wider range at reset:
  - `ue_relative_x ~ Uniform(0, 0.8 * inter_relay_dist)`

## Baseline Selection Rule

Threshold candidates are first filtered by attempt validity:
- `total_ho_attempted >= BASELINE_MIN_TOTAL_ATTEMPTS`

If valid candidates exist:
- select by priority:
  1. higher `aggregate_ehr`
  2. lower `aggregate_hfr`
  3. lower `aggregate_ppr`
  4. higher `mean_reward`
- `selection_mode = filtered_kpi_priority`

If none is valid:
- fallback to highest reward
- `selection_mode = fallback_reward`

## Diagnostics in Outputs

Baseline and training summaries include totals for:
- `total_ho_attempted`
- `total_ho_success`
- `total_ho_failed`
- `total_pingpong`
- `total_weak_signal_event`
- `no_attempt_episode_count`
- `no_attempt_episode_ratio`
- `total_pending_validation_started`
- `total_validation_success`
- `total_validation_failure`

## Run Commands

Highway baseline:
```bash
python 01_baseline_highway.py
```

Urban baseline:
```bash
python 01_baseline_urban.py
```

Highway training:
```bash
python 02_train_highway.py
```

Urban training:
```bash
python 02_train_urban.py
```

Plot figures:
```bash
python 03_plot_paper_figures.py --scenario highway
python 03_plot_paper_figures.py --scenario urban
```

## Figure Outputs (fixed)

- `fig_reward_trend.png`
- `fig_hfr_trend.png`
- `fig_ppr_trend.png`
- `fig_ehr_trend.png`
- `fig_hfr_bar.png`
- `fig_ppr_bar.png`
- `fig_ehr_bar.png`

Trend curves keep NaN semantics for no-attempt episodes, so they better reflect validation-based HO dynamics instead of instant-success artifacts.
