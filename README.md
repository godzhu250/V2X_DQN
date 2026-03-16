# V2X_DQN (Single-Scenario Pipeline)

This project uses a single-scenario workflow for SL-RSRP threshold adaptation:
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
- Each episode = `500 * 0.1 = 50 seconds`

Episode termination is fixed-length only.

## KPI Definitions

Per episode:
- `HFR = ho_failed / max(ho_attempted, 1)`
- `PPR = pingpong / max(ho_attempted, 1)`
- `EHR = 1 - HFR - PPR`

Final comparison uses aggregate KPI:
- `aggregate_HFR = sum(ho_failed) / sum(ho_attempted)`
- `aggregate_PPR = sum(pingpong) / sum(ho_attempted)`
- `aggregate_EHR = 1 - aggregate_HFR - aggregate_PPR`

## Reselection Logic

Attempt is triggered by:
- serving RSRP below selected threshold
- condition holds for TTT

Then success/failure is judged in a second stage:
- target relay quality above `TARGET_RSRP_FAIL_THRESHOLD_DBM`
- and target has at least `SUCCESS_MARGIN_DB` superiority over current serving relay
- `ho_attempted` is counted regardless of success/failure
- `HYSTERESIS_DB` is kept in config for compatibility/other logic, but success decision uses `SUCCESS_MARGIN_DB`

So target condition is no longer a pre-filter for whether attempt exists.

Ping-pong is defined as:
- a successful HO that returns to the previous relay
- within `PINGPONG_WINDOW_STEPS`

The current environment is still a simplified dual-relay environment.
It keeps the original dual-relay geometric flip backbone and adds explicit relay IDs
for stricter ping-pong detection consistency.

## Baseline Workflow

Highway:
```bash
python 01_baseline_highway.py
```

Urban:
```bash
python 01_baseline_urban.py
```

## Training Workflow

Shared core:
- `02_main.py` provides `run_training(scenario, seeds, episodes, output_dir)`

Highway entry (default 3 seeds + 3000 episodes):
```bash
python 02_train_highway.py
```

Urban entry:
```bash
python 02_train_urban.py
```

## Plotting Workflow

Highway:
```bash
python 03_plot_paper_figures.py --scenario highway
```

Urban:
```bash
python 03_plot_paper_figures.py --scenario urban
```

## Figure Outputs (fixed set)

Trend figures:
- `fig_reward_trend.png`
- `fig_hfr_trend.png`
- `fig_ppr_trend.png`
- `fig_ehr_trend.png`

Final bar figures:
- `fig_hfr_bar.png`
- `fig_ppr_bar.png`
- `fig_ehr_bar.png`

Plot colors:
- DQN (MLP): red
- LSTM-DQN: blue
- Fixed baseline: gray dashed line (trend)
