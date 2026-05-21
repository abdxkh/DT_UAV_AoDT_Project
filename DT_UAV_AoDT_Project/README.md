# DT-UAV AoDT Project

This repository is a clean coding structure for the **Age of Digital Twin (AoDT)** version of the DT-assisted UAV edge-network project.

## Main idea

- **Metric:** Age of Digital Twin (AoDT), not generic AoI.
- **Slow timescale manager:** chooses UAV hovering grid positions and DT placement every `H` slots.
- **Fast timescale worker:** chooses TDMA scheduling and uplink power level every slot.
- **Constraint:** long-term per-UAV backhaul energy budget.
- **Method:** Lyapunov-guided hierarchical PPO with vector virtual queue `Z_m`, one queue per UAV.

## Project structure

```text
DT_UAV_AoDT_Project/
  src/dt_uav/
    config.py          # configs: tiny, small, paper
    env.py             # AoDT simulator + manager wrapper
    ppo.py             # actor-critic PPO utilities
    train_worker.py    # fast-timescale worker PPO training
    train_manager.py   # slow-timescale manager PPO training with Lyapunov queue
    baselines.py       # random, k-means, AoDT-greedy, candidate-site search
    evaluate.py        # evaluation script
    quickstart.py      # tiny smoke test
  notebooks/
    run_colab.ipynb    # Colab runner notebook
  docs/
    DT_UAV_paper.pdf
    meeting_prep_conversation.md
  results/
```

## Colab usage

After pushing this folder to GitHub, in Colab run:

```python
!git clone https://github.com/YOUR_USERNAME/DT_UAV_AoDT_Project.git
%cd DT_UAV_AoDT_Project
!pip install -r requirements.txt
```

Then run the small demo:

```python
!PYTHONPATH=src python -m dt_uav.run_final --profile small --out results_colab --episodes 5 --candidate
```

This will train the worker and manager, run the baseline comparison, save `evaluation.csv`, and write plots into `results_colab/figures`.

For a very small smoke test:

```python
!PYTHONPATH=src python -m dt_uav.quickstart
```

For the small research setup with the separate steps:

```python
!PYTHONPATH=src python -m dt_uav.train_worker --profile small --out results_small
!PYTHONPATH=src python -m dt_uav.train_manager --profile small --out results_small --worker results_small/worker.pt
!PYTHONPATH=src python -m dt_uav.evaluate --profile small --out results_small --episodes 5 --candidate
```

For the paper-size setup later:

```python
!PYTHONPATH=src python -m dt_uav.train_worker --profile paper --out results_paper
!PYTHONPATH=src python -m dt_uav.train_manager --profile paper --out results_paper --worker results_paper/worker.pt
!PYTHONPATH=src python -m dt_uav.evaluate --profile paper --out results_paper --episodes 10
```

## Important notes

1. The code uses **AoDT naming everywhere**: `aodt_i`, `entity_aodt`, `avg_aodt`, `p95_aodt`, `max_aodt`.
2. UAVs are **piecewise-stationary**: fixed during each manager window and redeployed between windows.
3. The Lyapunov queue is for **backhaul energy**, matching the paper-aligned formulation.
4. The worker does **not** directly use the Lyapunov queue in its reward. The manager owns long-term constraint enforcement.
5. Candidate-site search is included as a stronger optimized baseline for small instances.

## Recommended development order

1. Run `quickstart` to confirm imports and code flow.
2. Run `small` profile to debug AoDT, energy, and queue behavior.
3. Tune `E_bh_max_per_uav` until the constraint is binding but feasible.
4. Compare against `candidate-site search` on small instances.
5. Scale to `paper` profile.
