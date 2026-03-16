# V2X_DQN (Single-Scenario Refactor)

This project now uses a clean single-scenario DRL pipeline for SL-RSRP threshold adaptation:
- Highway-only experiment flow
- Urban-only experiment flow

The old mixed scene switching flow (Highway/Urban alternating every 200 episodes) is removed from the main workflow.

## Project Goal

Train and compare:
- DQN (MLP)
- DRQN (LSTM)

against fixed-threshold baselines under one scenario at a time.

## Main Files

- `config.py`: global settings (scenario profiles, episode length, reward weights, training defaults)
- `vehicle_env.py`: single-scenario environment with fixed-length episode and attempt-based KPI accounting
- `base_agent.py`: agent interface
- `dqn_agent.py`: DQN (MLP)
- `lstm_dqn_agent.py`: DRQN (LSTM)
- `01_baseline_highway.py`: Highway fixed-threshold baseline search
- `01_baseline_urban.py`: Urban fixed-threshold baseline search
- `02_train_highway.py`: Highway training for DQN + DRQN
- `02_train_urban.py`: Urban training for DQN + DRQN
- `03_plot_paper_figures.py`: unified plotting script for paper/PPT-aligned figures

## Episode Time Definition

Each episode is fixed:
- `EPISODE_STEPS = 500`
- `SIM_STEP_SECONDS = 0.1`
- Episode duration = `500 * 0.1 = 50 seconds`

Episode termination is only based on step count reaching 500.

## KPI Definitions (Attempt-Based)

Per episode:
- `HFR = ho_failed / max(ho_attempted, 1)`
- `PPR = pingpong / max(ho_attempted, 1)`
- `EHR = 1 - HFR - PPR`

Notes:
- `ho_failed` counts only handover/reselection failures after attempt.
- `weak_signal_event` is tracked separately and does **not** count as HFR.

## Output Layout

- `Result/highway/baseline/...`
- `Result/highway/train/...`
- `Result/highway/figures/...`
- `Result/urban/baseline/...`
- `Result/urban/train/...`
- `Result/urban/figures/...`

## Highway Workflow

1. Run Highway baseline:

```bash
python 01_baseline_highway.py
```

2. Run Highway training (default: 3 seeds `71,123,456`, 3000 episodes):

```bash
python 02_train_highway.py
```

3. Generate Highway paper figures:

```bash
python 03_plot_paper_figures.py --scenario highway
```

## Urban Workflow

1. Run Urban baseline:

```bash
python 01_baseline_urban.py
```

2. Run Urban training:

```bash
python 02_train_urban.py
```

3. Generate Urban paper figures:

```bash
python 03_plot_paper_figures.py --scenario urban
```

## Deprecated Old Flow

The following old mixed-scene scripts are deprecated and removed from main usage:
- `01_discovery_and_baseline.py`
- `02_main.py`
- `02_seed_supplement.py`
- `03_kpi_trends.py`
- `04_bar_and_leap.py`
- `05_scene_bar.py`
- `06.py`
- `07.py`
