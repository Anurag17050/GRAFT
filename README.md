# GRAFT — Game-Theoretic Reinforcement Learning for Adaptive Fraud Detection and Trust-Aware Routing

> **Under Review** | CSE Department, Visvesvaraya National Institute of Technology (VNIT) Nagpur

---

## Overview

Decentralised peer-to-peer networks face two tightly coupled security problems: routing through untrustworthy intermediate nodes, and identifying fraudulent destination nodes in real time. Existing approaches handle these separately — routing protocols like AODV pick paths by hop count and ignore trust entirely, while fraud detectors classify nodes statically without influencing the route taken.

**GRAFT solves both problems jointly in a single framework.**

Each routing hop is modelled as a repeated **Prisoner's Dilemma**, where normalised trust scores define cooperation payoffs. A persistent **Q-learning agent** seeks to maximise cumulative path utility, converging to a **Nash-equilibrium routing policy**. Simultaneously, a **four-signal ensemble classifier** screens each destination node in real time, updating continuously from live RL experience — no retraining or labelled supervision required after deployment.

---

## Key Results

| Metric | Value |
|---|---|
| Fraud detection accuracy | **99.6%** across 5,858 classifiable nodes |
| Recall (fraud detection) | **100%** — zero missed malicious nodes |
| Routing success at deployment | **100%** (up from 50% during training) |
| Dataset | Bitcoin OTC Signed Trust Network — 5,881 nodes, 35,592 edges |
| Baselines outperformed | GCN, REV2, AODV, static XGBoost/RF classifiers |

---

## System Architecture

```
Bitcoin OTC CSV
      │
      ▼
┌─────────────────────────────────────────────────┐
│              GRAPH LOADER & PREPROCESSOR        │
│  Trust normalisation: raw[-10,+10] → [0.0,1.0]  │
│  Builds: adjacency list, edge trust map,        │
│          incoming/outgoing score tables         │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────┐        ┌────────────────────────────┐
│  PART 1      │        │  PART 2                    │
│  SAFE ROUTING│        │  DESTINATION CLASSIFICATION│
│              │        │                            │
│  BFS (AODV)  │        │  Four-Signal Ensemble      │
│  Dijkstra    │        │  Signal 1 (45%): avg_in    │
│  Q-Learning ─┼──────▶ │  Signal 2 (30%): fraud_   │
│  (Persistent)│        │              vote ratio    │
└──────┬───────┘        │  Signal 3 (15%): neg_ratio │
       │                │  Signal 4 (10%): avg_out   │
       ▼                │  + Live RL trust blend     │
  Q-table.json          └────────────────────────────┘
  phase.json
  stats.json
  node_reputation.json
```

---

## Core Components

### 1. Game-Theoretic Routing Model
Every hop in the route is modelled as a **non-cooperative repeated Prisoner's Dilemma**:
- Normalised trust score → cooperation payoff
- Malicious edge → defection penalty
- The Q-table converges to a Nash-equilibrium routing strategy, where deviating from the learned policy cannot improve expected cumulative reward

### 2. Persistent Q-Learning with Phase Separation

```
TRAIN phase (5 runs × 200 episodes):
  epsilon = 0.50  →  50% explore + 50% exploit
  Q-table grows with each run (JSON persistence)
  Success rate: ~50–93%

TEST / DEPLOYMENT phase (auto-switch after 5 trains):
  epsilon = 0.00  →  100% exploit, zero randomness
  Success rate: 95–100% consistently
```

**Reward structure:**

| Event | Reward |
|---|---|
| Reach destination | +20 |
| Safe intermediate hop | −1 (encourages shorter paths) |
| Malicious edge encountered | −10 (permanent blacklist) |

The Q-table persists across all sessions via `q_table.json`. Phase state is tracked automatically per source–destination pair — no manual input needed.

### 3. Four-Signal Ensemble Fraud Classifier

Each destination node is scored using four engineered features derived from the Bitcoin OTC dataset:

| Signal | Weight | Description |
|---|---|---|
| `avg_in` | 45% | Average trust score received from all raters |
| `fraud_ratio` | 30% | Fraction of raters who gave trust < 0.25 (explicit fraud flag) |
| `neg_ratio` | 15% | Fraction of raters who gave trust < 0.5 |
| `avg_out` | 10% | Average trust scores this node gave others (behavioural signal) |

Final fraud score = `(1 − avg_in) × 0.45 + fraud_ratio × 0.30 + neg_ratio × 0.15 + (1 − avg_out) × 0.10`

### 4. Live RL Reputation Model

Every node visited during routing episodes updates a live reputation score based on encounter outcomes:

| Encounter type | Effect on `rl_trust` |
|---|---|
| Safe pass | Small positive signal |
| Destination reached | Strong positive signal + bonus |
| Malicious drop | Sharp negative signal |

Prior (dataset) and live (RL) scores are blended dynamically:

| RL encounters | Blend |
|---|---|
| < 5 | 100% dataset (not enough live data) |
| 5–19 | 60% dataset + 40% live RL |
| ≥ 20 | 40% dataset + 60% live RL |

### 5. Convergence Analysis

After each run, a rolling window over the last 5 runs detects whether the agent has converged, is still improving, or hit a topology failure (all paths blocked by malicious nodes — which is itself a security success).

---

## Dataset

**Bitcoin OTC Signed Trust Network**
- Source: https://snap.stanford.edu/data/soc-sign-bitcoinotc.html
- Size: ~988 KB
- Nodes: 5,881 | Edges: 35,592
- Format: `source, target, trust (−10 to +10), timestamp`

Download the CSV and place it in the project root as `soc-sign-bitcoinotc.csv`.

Trust values are normalised on load:
```
norm = (raw_trust + 10) / 20.0
# −10 → 0.00 (scammer)
#   0 → 0.50 (neutral)
# +10 → 1.00 (fully trusted)
```

---

## Installation

GRAFT has **no external dependencies** — it uses only the Python standard library.

```bash
# Clone the repository
git clone https://github.com/Anurag17050/GRAFT.git
cd GRAFT

# Download the dataset
# Visit: https://snap.stanford.edu/data/soc-sign-bitcoinotc.html
# Place soc-sign-bitcoinotc.csv in the project root

# Run
python3 final.py
```

**Requirements:** Python 3.7+, no pip installs needed.

---

## Usage

```bash
python3 final.py
```

You will be prompted for a source node and destination node. The simulator then:
1. Loads the graph and any saved Q-table
2. Determines current phase (TRAIN or TEST) for this pair
3. Runs BFS and Dijkstra for comparison baselines
4. Runs Q-learning episodes (200 in TRAIN, 100 in TEST)
5. Classifies the destination node using the four-signal ensemble
6. Prints a full comparison table and learning progress

**Run the same pair multiple times** to observe the Q-table growing and success rate improving. After 5 training runs, the system automatically switches to TEST mode (epsilon = 0.0).

To start completely fresh:
```bash
rm q_table.json stats.json phase.json node_reputation.json
```

### Recommended Test Pairs

| Pair | Expected Behaviour |
|---|---|
| `7 → 65` | Moderate difficulty. ~78% on run 1, ~95% by run 3. Best proof of persistent RL improvement. |
| `1317 → 1` | Easy path. ~100% from run 1. Demonstrates pure trust-path routing. |
| `4135 → 4182` | Topology failure. 0% success — every path contains a malicious node. Demonstrates security enforcement over blind delivery. |

---

## Configuration

All hyperparameters are at the top of `final.py`:

```python
TRAIN_RUNS     = 5      # training runs before auto-switching to TEST
TRAIN_EPISODES = 200    # RL episodes per training run
EPSILON_TRAIN  = 0.50   # exploration rate during training
TEST_EPISODES  = 100    # RL episodes per test run
EPSILON_TEST   = 0.00   # pure exploitation at deployment
EPSILON_MIN    = 0.10   # minimum epsilon floor
EPSILON_DECAY  = 0.95   # per-episode epsilon decay
LR             = 0.50   # Q-learning rate (α)
GAMMA          = 0.90   # discount factor
DROP_THRESHOLD = 0.25   # edges below this trust are malicious
MAX_STEPS      = 80     # maximum hops per episode
```

---

## Persistent Files

| File | Contents |
|---|---|
| `q_table.json` | Serialised Q-table — edge-level (state, action) values. Grows across all runs. |
| `phase.json` | Tracks how many training runs have been completed per source–destination pair. |
| `stats.json` | Full run history — success rates, episode counts, Q-table sizes, phase labels. |
| `node_reputation.json` | Live RL reputation — encounter counts and learned trust per node visited during routing. |

---

## Comparison with Baselines

| Method | Routing intelligence | Fraud detection | Adapts over time |
|---|---|---|---|
| AODV (BFS) | Hop count only | None | No |
| Dijkstra (trust-only) | Max avg trust | None | No |
| Static ML (XGBoost/RF) | None | Yes (offline) | No — requires full retrain |
| GCN / REV2 | None | Yes (graph-structural) | No — static predictors |
| **GRAFT (ours)** | **Nash-equilibrium Q-routing** | **Yes (real-time, 4-signal)** | **Yes — live RL updates** |

---

## Results Summary

```
Pair 7 → 65 (proof of improvement):
  Run 1 (TRAIN): 78.0%
  Run 3 (TRAIN): 95.0%
  Run 6+ (TEST): 100.0%

Full network evaluation:
  Classifiable nodes : 5,858
  Fraud accuracy     : 99.6%
  Recall             : 100.0%  (zero missed fraud nodes)
  Routing success    : 100.0% at deployment
```

---

## File Structure

```
GRAFT/
├── final.py                    # Complete simulator — routing + fraud detection
├── soc-sign-bitcoinotc.csv     # Dataset (download separately)
├── q_table.json                # Auto-generated: persisted Q-table
├── phase.json                  # Auto-generated: train/test phase tracker
├── stats.json                  # Auto-generated: run history
├── node_reputation.json        # Auto-generated: live RL reputation
└── README.md
```

---

## How It Works — End to End

```
User inputs src → dst
        │
        ├─► BFS finds minimum-hop path (AODV baseline)
        ├─► Dijkstra finds max-trust path (trust baseline)
        │
        ├─► Q-agent runs N episodes:
        │     For each episode:
        │       At each hop → epsilon-greedy action selection
        │       If malicious edge → reward −10, blacklist, end episode
        │       If safe hop      → reward −1, update Q, update reputation
        │       If destination   → reward +20, success
        │     Q-table updated online after every single hop
        │     Node reputation updated for every node visited
        │
        ├─► Phase manager auto-switches TRAIN → TEST after 5 runs
        │
        ├─► Exploit path extracted from converged Q-table
        │
        └─► Destination classified:
              Four signals computed from dataset
              Blended with live RL reputation
              → HIGHLY TRUSTED / TRUSTED / SUSPICIOUS / FRAUD
```

---

## Authors

- **Anurag Marda** — CSE, VNIT Nagpur
- **Supraja Soudu** — CSE, VNIT Nagpur
- **Anshul Agarwal** — CSE, VNIT Nagpur

---

## Citation

If you use this work, please cite:

```bibtex
@article{marda2025graft,
  title   = {GRAFT: Game-Theoretic Reinforcement Learning for Adaptive
             Fraud Detection and Trust-Aware Routing},
  author  = {Marda, Anurag and Soudu, Supraja and Agarwal, Anshul},
  journal = {Under Review},
  year    = {2025},
  note    = {CSE Department, VNIT Nagpur}
}
```

---

## License

This repository is shared for academic and research purposes. Please contact the authors before using this work in any commercial application.
