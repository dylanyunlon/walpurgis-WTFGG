# Walpurgis — Workload-Aware GNN Training on Mixed-Generation GPUs

> *"Die Hexen zu dem Brocken ziehn, die Stoppel ist gelb, die Saat ist grün."*
> — Walpurgisnacht, Faust I

A thousand demons with varied powers converge on one summit —
heterogeneous GPUs with asymmetric compute must coordinate GNN training.
Big batches → H100, small batches + neighbor sampling → A6000, feature preprocessing → CPU.

## Upstream

| Directory | Origin | Role |
|-----------|--------|------|
| `upstream/morphgl` | [initzhang/MorphGL](https://github.com/initzhang/MorphGL) | Collective batching & scheduling for GNN |
| `upstream/d2stgnn` | [GestaltCogTeam/D2STGNN](https://github.com/GestaltCogTeam/D2STGNN) | Decoupled spatial-temporal graph neural network |
