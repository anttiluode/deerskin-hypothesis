"""
Benchmark: MoiréNet vs MLP on XOR
==================================
Compares geometry-based computation (moiré interference)
against standard scalar-weight neural networks.

Both use exactly 9 parameters. Same task. Different substrate.

Part of the Deerskin Hypothesis project.
"""

import numpy as np
import time
from moire_net import MoireNet

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([0, 1, 1, 0], dtype=float)


def simple_mlp_xor(max_epochs=1000, lr=0.5):
    """
    Train a minimal MLP on XOR.
    Architecture: 2-2-1 with sigmoid activation.
    Parameters: 2×2 + 2 + 2×1 + 1 = 9 (same as MoiréNet)
    """
    # Random init
    W1 = np.random.randn(2, 2) * 0.5
    b1 = np.zeros(2)
    W2 = np.random.randn(2, 1) * 0.5
    b2 = np.zeros(1)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    for epoch in range(max_epochs):
        # Forward
        h = sigmoid(X @ W1 + b1)
        o = sigmoid(h @ W2 + b2).flatten()

        # Loss
        loss = np.mean((o - Y) ** 2)

        # Check accuracy
        acc = np.mean((o > 0.5) == Y)
        if acc >= 1.0:
            return 1.0, epoch + 1

        # Backward (manual gradients)
        do = (o - Y) * o * (1 - o)
        dW2 = h.T @ do.reshape(-1, 1)
        db2 = np.sum(do)
        dh = do.reshape(-1, 1) @ W2.T * h * (1 - h)
        dW1 = X.T @ dh
        db1 = np.sum(dh, axis=0)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    # Final accuracy
    h = sigmoid(X @ W1 + b1)
    o = sigmoid(h @ W2 + b2).flatten()
    return float(np.mean((o > 0.5) == Y)), max_epochs


def run_benchmark(n_trials=30):
    print("=" * 65)
    print("  BENCHMARK: MoiréNet vs MLP on XOR")
    print("  Both use 9 parameters. Same task. Different substrate.")
    print("=" * 65)
    print()

    # MoiréNet trials
    moire_results = []
    moire_times = []
    print(f"Running {n_trials} MoiréNet trials...")
    for i in range(n_trials):
        net = MoireNet(n_hidden=3, grid_size=32)
        t0 = time.time()
        acc, gens = net.evolve_for_xor(generations=200, population=50)
        elapsed = time.time() - t0
        moire_results.append((acc, gens))
        moire_times.append(elapsed)
        status = "✓" if acc >= 1.0 else "✗"
        print(f"  [{status}] Trial {i+1:2d}: acc={acc:.0%}, gens={gens:4d}, time={elapsed:.3f}s")

    # MLP trials
    mlp_results = []
    mlp_times = []
    print(f"\nRunning {n_trials} MLP trials...")
    for i in range(n_trials):
        t0 = time.time()
        acc, epochs = simple_mlp_xor(max_epochs=1000)
        elapsed = time.time() - t0
        mlp_results.append((acc, epochs))
        mlp_times.append(elapsed)
        status = "✓" if acc >= 1.0 else "✗"
        print(f"  [{status}] Trial {i+1:2d}: acc={acc:.0%}, epochs={epochs:4d}, time={elapsed:.3f}s")

    # Summary
    moire_solved = sum(1 for a, _ in moire_results if a >= 1.0)
    moire_gens = [g for a, g in moire_results if a >= 1.0]
    mlp_solved = sum(1 for a, _ in mlp_results if a >= 1.0)
    mlp_epochs = [e for a, e in mlp_results if a >= 1.0]

    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print()
    print(f"  {'Metric':<25} {'MoiréNet':>15} {'MLP':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Parameters':<25} {'9 (geometric)':>15} {'9 (scalar)':>15}")
    print(f"  {'Solved':<25} {f'{moire_solved}/{n_trials}':>15} {f'{mlp_solved}/{n_trials}':>15}")

    if moire_gens:
        print(f"  {'Mean convergence':<25} {f'{np.mean(moire_gens):.0f} gens':>15}", end="")
    else:
        print(f"  {'Mean convergence':<25} {'N/A':>15}", end="")

    if mlp_epochs:
        print(f" {f'{np.mean(mlp_epochs):.0f} epochs':>15}")
    else:
        print(f" {'N/A':>15}")

    print(f"  {'Mean time':<25} {f'{np.mean(moire_times):.3f}s':>15} {f'{np.mean(mlp_times):.3f}s':>15}")
    print()

    if moire_solved > mlp_solved:
        print("  → MoiréNet wins: higher solve rate with geometric parameters")
    elif moire_solved == mlp_solved:
        print("  → Tied on solve rate")
    else:
        print("  → MLP wins on solve rate")

    print()


if __name__ == "__main__":
    run_benchmark(30)
