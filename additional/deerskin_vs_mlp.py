"""
Deerskin vs MLP: Multi-Task Benchmark
======================================
Tests the core claim: geometric parameters encode more structure
than scalar weights at the same parameter count.

Tasks: XOR, AND, OR, NAND, Frequency Separation
Metrics: Solve rate, convergence speed, noise robustness

Run: python deerskin_vs_mlp.py
"""

import numpy as np
import time
import sys

# ================================================================
# MOIRE NET (Geometric substrate)
# ================================================================

class MoireNeuron:
    def __init__(self, freq, angle, phase, grid_size=32):
        self.freq = freq
        self.angle = angle
        self.phase = phase
        self.gs = grid_size
        self._build()

    def _build(self):
        s = self.gs
        y, x = np.mgrid[0:s, 0:s]
        c, sn = np.cos(self.angle), np.sin(self.angle)
        rx = (x - s/2) * c - (y - s/2) * sn + self.phase
        ry = (x - s/2) * sn + (y - s/2) * c
        cell = max(1, s / (self.freq + 1e-6))
        gx = np.floor(rx / cell).astype(int) % 2
        gy = np.floor(ry / cell).astype(int) % 2
        self.grid = (gx ^ gy).astype(np.float32)

    def set(self, f, a, p):
        self.freq, self.angle, self.phase = f, a, p
        self._build()


class MoireNet:
    def __init__(self, n_hidden=3, gs=32):
        self.neurons = [
            MoireNeuron(np.random.uniform(3, 25),
                        np.random.uniform(0, np.pi),
                        np.random.uniform(0, 2*np.pi), gs)
            for _ in range(n_hidden)
        ]
        self.gs = gs
        self.n_hidden = n_hidden

    def forward(self, X):
        X = np.atleast_2d(X)
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            px = int(np.clip(x[0] * (self.gs - 1), 0, self.gs - 1))
            py = int(np.clip(x[1] * (self.gs - 1), 0, self.gs - 1))
            vals = [n.grid[py, px] for n in self.neurons]
            out[i] = sum(vals) % 2
        return out

    def get_params(self):
        p = []
        for n in self.neurons:
            p.extend([n.freq, n.angle, n.phase])
        return np.array(p)

    def set_params(self, p):
        for i, n in enumerate(self.neurons):
            n.set(p[i*3], p[i*3+1] % np.pi, p[i*3+2] % (2*np.pi))

    def evolve(self, X, Y, generations=300, pop_size=60):
        best = self.get_params()
        best_acc = 0
        for g in range(generations):
            candidates = [best + np.random.randn(len(best)) * 0.4 for _ in range(pop_size)]
            for c in candidates:
                for j in range(self.n_hidden):
                    c[j*3] = np.clip(c[j*3], 1, 30)
                self.set_params(c)
                pred = self.forward(X)
                acc = np.mean((pred > 0.5) == Y)
                if acc > best_acc:
                    best_acc = acc
                    best = c.copy()
            if best_acc >= 1.0:
                self.set_params(best)
                return 1.0, g + 1
        self.set_params(best)
        return best_acc, generations


# ================================================================
# MLP (Scalar weight substrate)
# ================================================================

class SimpleMLP:
    """2-2-1 MLP with sigmoid. 9 parameters: W1(2x2)=4, b1(2)=2, W2(2x1)=2, b2(1)=1"""
    def __init__(self):
        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.zeros(2)
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.zeros(1)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, X):
        h = self.sigmoid(X @ self.W1 + self.b1)
        return self.sigmoid(h @ self.W2 + self.b2).flatten()

    def train(self, X, Y, max_epochs=1000, lr=0.5):
        for ep in range(max_epochs):
            h = self.sigmoid(X @ self.W1 + self.b1)
            o = self.sigmoid(h @ self.W2 + self.b2).flatten()
            acc = np.mean((o > 0.5) == Y)
            if acc >= 1.0:
                return 1.0, ep + 1
            do = (o - Y) * o * (1 - o)
            dW2 = h.T @ do.reshape(-1, 1)
            db2 = np.sum(do)
            dh = do.reshape(-1, 1) @ self.W2.T * h * (1 - h)
            dW1 = X.T @ dh
            db1 = np.sum(dh, axis=0)
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
        h = self.sigmoid(X @ self.W1 + self.b1)
        o = self.sigmoid(h @ self.W2 + self.b2).flatten()
        return float(np.mean((o > 0.5) == Y)), max_epochs


# ================================================================
# BENCHMARK TASKS
# ================================================================

TASKS = {
    'XOR':  (np.array([[0,0],[0,1],[1,0],[1,1]], float),
             np.array([0,1,1,0], float)),
    'AND':  (np.array([[0,0],[0,1],[1,0],[1,1]], float),
             np.array([0,0,0,1], float)),
    'OR':   (np.array([[0,0],[0,1],[1,0],[1,1]], float),
             np.array([0,1,1,1], float)),
    'NAND': (np.array([[0,0],[0,1],[1,0],[1,1]], float),
             np.array([1,1,1,0], float)),
}


def run_task(name, X, Y, n_trials=20):
    """Run both architectures on a task, collect statistics."""
    moire_solved, moire_gens, moire_times = 0, [], []
    mlp_solved, mlp_epochs, mlp_times = 0, [], []

    for _ in range(n_trials):
        # MoiréNet
        net = MoireNet(3, 32)
        t0 = time.time()
        acc, g = net.evolve(X, Y)
        moire_times.append(time.time() - t0)
        if acc >= 1.0:
            moire_solved += 1
            moire_gens.append(g)

        # MLP
        mlp = SimpleMLP()
        t0 = time.time()
        acc, e = mlp.train(X, Y)
        mlp_times.append(time.time() - t0)
        if acc >= 1.0:
            mlp_solved += 1
            mlp_epochs.append(e)

    return {
        'task': name,
        'moire_rate': moire_solved / n_trials,
        'moire_conv': np.mean(moire_gens) if moire_gens else float('inf'),
        'moire_time': np.mean(moire_times),
        'mlp_rate': mlp_solved / n_trials,
        'mlp_conv': np.mean(mlp_epochs) if mlp_epochs else float('inf'),
        'mlp_time': np.mean(mlp_times),
    }


def noise_robustness_test(n_trials=20):
    """Test how both architectures degrade under input noise."""
    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    Y_xor = np.array([0,1,1,0], float)
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

    print("\n  NOISE ROBUSTNESS (XOR, trained clean, tested noisy)")
    print(f"  {'Noise':>8} {'MoiréNet':>12} {'MLP':>12}")
    print(f"  {'─'*8} {'─'*12} {'─'*12}")

    for noise in noise_levels:
        moire_accs, mlp_accs = [], []

        for _ in range(n_trials):
            # Train both clean
            net = MoireNet(3, 32)
            acc_m, _ = net.evolve(X_xor, Y_xor)
            if acc_m < 1.0:
                continue

            mlp = SimpleMLP()
            acc_s, _ = mlp.train(X_xor, Y_xor)
            if acc_s < 1.0:
                continue

            # Test noisy (100 noisy samples per pattern)
            correct_m, correct_s, total = 0, 0, 0
            for xi, yi in zip(X_xor, Y_xor):
                for _ in range(100):
                    noisy = xi + np.random.randn(2) * noise
                    noisy = np.clip(noisy, -0.5, 1.5)

                    pm = net.forward(noisy.reshape(1, -1))[0]
                    ps = mlp.forward(noisy.reshape(1, -1))[0]

                    if (pm > 0.5) == yi:
                        correct_m += 1
                    if (ps > 0.5) == yi:
                        correct_s += 1
                    total += 1

            if total > 0:
                moire_accs.append(correct_m / total)
                mlp_accs.append(correct_s / total)

        ma = np.mean(moire_accs) if moire_accs else 0
        sa = np.mean(mlp_accs) if mlp_accs else 0
        winner = "◀ Moiré" if ma > sa + 0.01 else ("◀ MLP" if sa > ma + 0.01 else "  tie")
        print(f"  {noise:8.2f} {ma:11.1%} {sa:11.1%}  {winner}")


# ================================================================
# MAIN
# ================================================================

def main():
    n_trials = 20

    print("=" * 65)
    print("  DEERSKIN vs MLP: Multi-Task Benchmark")
    print("  9 geometric params vs 9 scalar weights")
    print(f"  {n_trials} trials per task")
    print("=" * 65)

    results = []
    for name, (X, Y) in TASKS.items():
        sys.stdout.write(f"\n  Running {name}...")
        sys.stdout.flush()
        r = run_task(name, X, Y, n_trials)
        results.append(r)
        print(" done")

    # Summary table
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"\n  {'Task':<8} {'Moiré Rate':>12} {'Moiré Conv':>12} {'MLP Rate':>12} {'MLP Conv':>12} {'Winner':>8}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")

    moire_wins = 0
    mlp_wins = 0
    for r in results:
        mc = f"{r['moire_conv']:.0f}g" if r['moire_conv'] < 300 else "—"
        sc = f"{r['mlp_conv']:.0f}e" if r['mlp_conv'] < 1000 else "—"

        if r['moire_rate'] > r['mlp_rate'] + 0.05:
            winner = "Moiré"
            moire_wins += 1
        elif r['mlp_rate'] > r['moire_rate'] + 0.05:
            winner = "MLP"
            mlp_wins += 1
        else:
            winner = "tie"

        print(f"  {r['task']:<8} {r['moire_rate']:>11.0%} {mc:>12} {r['mlp_rate']:>11.0%} {sc:>12} {winner:>8}")

    print(f"\n  Score: MoiréNet {moire_wins} — MLP {mlp_wins}")

    # Noise test
    noise_robustness_test(n_trials=15)

    # Speed comparison
    print(f"\n  AVERAGE TIME PER TRIAL:")
    for r in results:
        print(f"    {r['task']}: Moiré {r['moire_time']:.3f}s  MLP {r['mlp_time']:.3f}s")

    print()


if __name__ == "__main__":
    main()
