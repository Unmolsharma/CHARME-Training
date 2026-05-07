# CHARME Training Pipeline v2

Faithful reconstruction of the CHARME training pipeline from:
> "CHARME: A Chain-based Reinforcement Learning Approach for the Minor Embedding Problem"
> Ngo et al., ACM Transactions on Quantum Computing, 2025

## What changed vs v1

| Component | v1 | v2 |
|---|---|---|
| `models.py` | Rewritten | Authors' exact code, only `evaluate()` signature fixed |
| `ppo.py` | Rewritten | Authors' exact code, `update()` globals fixed |
| `env.py` | Rewritten | Authors' exact code, paths made configurable |
| `train.py` | Rewritten | Authors' exact loop, `update()` call fixed |
| `02_generate_orderlist.py` | Random search approximation | Full Algorithm 2+3 with ATOM calls and Theorem 1 pruning |

## One required manual step

Copy the real `utils.py` from the charme-rl repo:
```bash
# Linux (lab machine)
cp ~/charme-rl/charme/utils.py ~/charme_training_v2/charme/utils.py

# Windows
copy C:\Users\unmol\algorithms\charme-rl\charme\utils.py charme\utils.py
```

## Paper-Exact Training Configuration

| Parameter | Value | Source |
|---|---|---|
| Training graphs | 70 BA graphs | Section 4.1 |
| Graph size | n=150, d=10 | Section 4.1 |
| Hardware | Chimera(45,45,4) | Section 4.1 |
| Actor lr | 3×10⁻⁴ | Section 4.1 |
| Critic lr | 1×10⁻⁵ | Section 4.1 |
| K_epochs | 80 | TrainConfig |
| Update every | 100 episodes | Section 4.1 |
| Batch size | 290 | Section 4.1 |
| Order Exploration D | 10,000 | Section 4.1 |
| Order Exploration K | 10⁶ | Section 4.1 |
| Threads | 1024 | Section 4.1 |

## Step-by-Step

### 0. Setup (on lab GPU machine)

```bash
conda create -n charme python=3.10 -y
conda activate charme
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric networkx karateclub minorminer dwave-networkx tqdm numpy
```

Compile ATOM:
```bash
cd ~/charme-rl/ours
g++ -O2 -o atom_system atom_system.cpp
```

Copy utils:
```bash
cp ~/charme-rl/charme/utils.py ~/charme_training_v2/charme/utils.py
```

### 1. Generate training graphs (~20-30 min)

```bash
cd ~/charme_training_v2
python scripts/01_generate_training_data.py \
    --hw_topo_row 45 --hw_topo_col 45 --hw_bipart_cell 4 \
    --lg_num_nodes 150 --lg_degree 10 --n_graphs 70
```

### 2. Generate orderlist — Algorithm 2+3 (hours)

```bash
python scripts/02_generate_orderlist.py \
    --hw_topo_row 45 --hw_topo_col 45 --hw_bipart_cell 4 \
    --sampling_limit 10000 --exploration_limit 1000000 \
    --n_threads 1024 \
    --atom_binary ~/charme-rl/ours/atom_system
```

Quick smoke test (minutes):
```bash
python scripts/02_generate_orderlist.py \
    --sampling_limit 50 --exploration_limit 100 --n_threads 4 \
    --atom_binary ~/charme-rl/ours/atom_system
```

### 3. Train (12-48h on GPU)

```bash
screen -S charme
python train.py \
    --atom_binary ~/charme-rl/ours/atom_system \
    --device cuda
```

Detach: `Ctrl+A D`
Reattach: `screen -r charme`

Checkpoints saved to `checkpoints/ppo_CHARME_N.pth` every 100k steps.

### 4. Copy to QEBench

```bash
cp checkpoints/ppo_CHARME_2000000.pth \
   ~/quantum_embedding_benchmark/algorithms/charme/checkpoints/charme_trained.pth
```

## What to expect during training

Per Figure 7 in the paper:

- **Without orderlist**: stuck at ~8200 avg qubits for 12,000 steps
- **With orderlist**: drops to ~7800 within 1,000 steps, ~6900 by step 9,000
- **After 30,000 steps**: ~25% reduction from baseline

The log prints every 100 episodes:
```
Episode    100 | Timestep     8,234 | Reward  -142.3 | RL qubits  7823.1 | MM qubits  7901.4 | sigma 0.00
Episode    200 | Timestep    16,471 | Reward  -118.7 | RL qubits  7512.3 | MM qubits  7901.4 | sigma 0.01
```

A good sign: RL qubits dropping below MM qubits.
