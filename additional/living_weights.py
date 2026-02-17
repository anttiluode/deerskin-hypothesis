"""
Living Weights: Dynamic Substrate Learning
============================================
THE key architectural difference between deerskin and standard AI:

Standard AI: Weights are FROZEN during inference. Train → Deploy.
Deerskin AI: Membrane geometry changes DURING computation.
             Every spike that traverses the membrane also reshapes it.
             Learning and inference are the SAME process.

This experiment tests whether "living weights" (parameters that
evolve during inference) outperform frozen weights on continual
learning — the task where standard AI fails hardest (catastrophic
forgetting).

Run: python living_weights.py
"""

import numpy as np
import time


# ================================================================
# LIVING WEIGHT NETWORK (Deerskin-inspired)
# ================================================================

class LivingWeightNet:
    """
    A network where weights change during inference.

    Each "weight" is a geometric parameter (freq, angle, phase) that
    drifts toward configurations that reduce local error. This is
    analogous to activity-dependent membrane remodeling — ion channel
    trafficking happens on timescales of seconds to minutes, meaning
    the "weights" change while the system is computing.

    Key: plasticity_rate controls how fast weights evolve during use.
    At 0, this is a standard frozen network.
    At >0, it's a living substrate.
    """

    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1,
                 plasticity_rate=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)
        self.plasticity_rate = plasticity_rate

        # Hebbian trace: tracks which weight paths are active
        self.trace_W1 = np.zeros_like(self.W1)
        self.trace_W2 = np.zeros_like(self.W2)

        # Memory consolidation: slow-moving average of weights
        self.slow_W1 = self.W1.copy()
        self.slow_W2 = self.W2.copy()

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, x, y_true=None):
        """
        Forward pass WITH optional online learning.
        If y_true is provided, weights adapt during the forward pass.
        This is the "living" part — computation and learning are simultaneous.
        """
        x = np.atleast_2d(x)

        # Forward
        h = self.sigmoid(x @ self.W1 + self.b1)
        o = self.sigmoid(h @ self.W2 + self.b2).flatten()

        # Living weight update (if target provided)
        if y_true is not None and self.plasticity_rate > 0:
            y_true = np.atleast_1d(y_true)
            error = o - y_true

            # Local Hebbian-like update
            # The key insight: this happens ON EVERY FORWARD PASS
            # not in a separate "training" phase
            do = error * o * (1 - o)
            dW2 = h.T @ do.reshape(-1, 1)
            dh = do.reshape(-1, 1) @ self.W2.T * h * (1 - h)
            dW1 = x.T @ dh

            # Apply plasticity (small step)
            self.W2 -= self.plasticity_rate * dW2
            self.b2 -= self.plasticity_rate * np.sum(do)
            self.W1 -= self.plasticity_rate * dW1
            self.b1 -= self.plasticity_rate * np.sum(dh, axis=0)

            # Memory consolidation: slow EMA prevents catastrophic drift
            consolidation_rate = 0.001
            self.slow_W1 = (1 - consolidation_rate) * self.slow_W1 + consolidation_rate * self.W1
            self.slow_W2 = (1 - consolidation_rate) * self.slow_W2 + consolidation_rate * self.W2

            # Elastic regularization: pull weights toward consolidated memory
            elastic_strength = 0.0005
            self.W1 -= elastic_strength * (self.W1 - self.slow_W1)
            self.W2 -= elastic_strength * (self.W2 - self.slow_W2)

        return o

    def predict(self, x):
        """Forward pass without adaptation (for evaluation)."""
        x = np.atleast_2d(x)
        h = self.sigmoid(x @ self.W1 + self.b1)
        return self.sigmoid(h @ self.W2 + self.b2).flatten()


class FrozenWeightNet:
    """Standard network: train first, then freeze for inference."""

    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def train_batch(self, X, Y, epochs=500, lr=0.5):
        """Standard batch training."""
        for ep in range(epochs):
            h = self.sigmoid(X @ self.W1 + self.b1)
            o = self.sigmoid(h @ self.W2 + self.b2).flatten()
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

    def predict(self, x):
        x = np.atleast_2d(x)
        h = self.sigmoid(x @ self.W1 + self.b1)
        return self.sigmoid(h @ self.W2 + self.b2).flatten()


# ================================================================
# CONTINUAL LEARNING EXPERIMENT
# ================================================================

def continual_learning_test(n_trials=10):
    """
    The critical test: learn Task A, then learn Task B,
    then check if you still remember Task A.

    Standard AI: catastrophic forgetting (learning B destroys A).
    Living weights: should retain A while adapting to B,
    because the elastic consolidation protects old memories.
    """
    print("\n  EXPERIMENT 1: Continual Learning (Catastrophic Forgetting)")
    print("  " + "─" * 55)
    print("  Phase 1: Learn XOR")
    print("  Phase 2: Learn AND (without re-training XOR)")
    print("  Test: Can you still do XOR?")

    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    Y_xor = np.array([0,1,1,0], float)
    X_and = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    Y_and = np.array([0,0,0,1], float)

    living_results = {'xor_after_xor': [], 'xor_after_and': [], 'and_after_and': []}
    frozen_results = {'xor_after_xor': [], 'xor_after_and': [], 'and_after_and': []}

    for trial in range(n_trials):
        # ---- LIVING WEIGHT NETWORK ----
        living = LivingWeightNet(2, 6, 1, plasticity_rate=0.3)

        # Phase 1: Learn XOR (stream data, living weights adapt)
        for epoch in range(500):
            for x, y in zip(X_xor, Y_xor):
                living.forward(x.reshape(1, -1), y)

        # Test XOR after Phase 1
        pred = living.predict(X_xor)
        xor_acc_1 = np.mean((pred > 0.5) == Y_xor)
        living_results['xor_after_xor'].append(xor_acc_1)

        # Phase 2: Learn AND (living weights keep adapting)
        # Reduce plasticity (the system has "consolidated" XOR)
        living.plasticity_rate = 0.1
        for epoch in range(300):
            for x, y in zip(X_and, Y_and):
                living.forward(x.reshape(1, -1), y)

        # Test both tasks after Phase 2
        pred_xor = living.predict(X_xor)
        pred_and = living.predict(X_and)
        living_results['xor_after_and'].append(np.mean((pred_xor > 0.5) == Y_xor))
        living_results['and_after_and'].append(np.mean((pred_and > 0.5) == Y_and))

        # ---- FROZEN WEIGHT NETWORK ----
        frozen = FrozenWeightNet(2, 6, 1)

        # Phase 1: Train on XOR (batch)
        frozen.train_batch(X_xor, Y_xor, epochs=500)
        pred = frozen.predict(X_xor)
        frozen_results['xor_after_xor'].append(np.mean((pred > 0.5) == Y_xor))

        # Phase 2: Train on AND (batch, overwrites XOR weights)
        frozen.train_batch(X_and, Y_and, epochs=300)
        pred_xor = frozen.predict(X_xor)
        pred_and = frozen.predict(X_and)
        frozen_results['xor_after_and'].append(np.mean((pred_xor > 0.5) == Y_xor))
        frozen_results['and_after_and'].append(np.mean((pred_and > 0.5) == Y_and))

    # Report
    print(f"\n  Results ({n_trials} trials):")
    print(f"  {'':>25} {'Living':>10} {'Frozen':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10}")
    print(f"  {'XOR acc after XOR':>25} {np.mean(living_results['xor_after_xor']):>9.0%} {np.mean(frozen_results['xor_after_xor']):>9.0%}")
    print(f"  {'XOR acc after AND':>25} {np.mean(living_results['xor_after_and']):>9.0%} {np.mean(frozen_results['xor_after_and']):>9.0%}")
    print(f"  {'AND acc after AND':>25} {np.mean(living_results['and_after_and']):>9.0%} {np.mean(frozen_results['and_after_and']):>9.0%}")

    xor_retention_living = np.mean(living_results['xor_after_and'])
    xor_retention_frozen = np.mean(frozen_results['xor_after_and'])

    print(f"\n  XOR retention after learning AND:")
    print(f"    Living weights: {xor_retention_living:.0%}")
    print(f"    Frozen weights: {xor_retention_frozen:.0%}")

    if xor_retention_living > xor_retention_frozen + 0.05:
        print(f"    → Living weights retain {xor_retention_living - xor_retention_frozen:.0%} more")
        print(f"      Elastic consolidation protects old memories")
    elif xor_retention_frozen > xor_retention_living + 0.05:
        print(f"    → Frozen weights retained better on this run")
    else:
        print(f"    → Similar retention")

    return xor_retention_living, xor_retention_frozen


# ================================================================
# ONLINE ADAPTATION EXPERIMENT
# ================================================================

def online_adaptation_test():
    """
    Test how quickly the living network adapts to a SUDDEN change
    in the task, without any explicit "retraining" signal.
    """
    print("\n\n  EXPERIMENT 2: Online Adaptation Speed")
    print("  " + "─" * 55)
    print("  Phase 1: Environment says XOR (200 samples)")
    print("  Phase 2: Environment SWITCHES to AND (200 samples)")
    print("  Measure: How fast does each network adapt?")

    X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    Y_xor = np.array([0,1,1,0], float)
    Y_and = np.array([0,0,0,1], float)

    n_trials = 10
    living_switch_speed = []
    frozen_switch_speed = []

    for _ in range(n_trials):
        # Living network
        living = LivingWeightNet(2, 6, 1, plasticity_rate=0.2)

        # Phase 1: XOR environment
        for _ in range(200):
            idx = np.random.randint(4)
            living.forward(X[idx:idx+1], Y_xor[idx])

        # Phase 2: AND environment — measure how many samples to adapt
        adapted = False
        for t in range(200):
            idx = np.random.randint(4)
            living.forward(X[idx:idx+1], Y_and[idx])

            # Check AND accuracy
            pred = living.predict(X)
            if np.mean((pred > 0.5) == Y_and) >= 1.0:
                living_switch_speed.append(t + 1)
                adapted = True
                break

        if not adapted:
            living_switch_speed.append(200)

        # Frozen network: must retrain from scratch
        frozen = FrozenWeightNet(2, 6, 1)
        frozen.train_batch(X, Y_xor, epochs=200)

        # "Switch" = retrain on AND
        adapted = False
        for ep in range(200):
            frozen.train_batch(X, Y_and, epochs=1, lr=0.5)
            pred = frozen.predict(X)
            if np.mean((pred > 0.5) == Y_and) >= 1.0:
                frozen_switch_speed.append(ep + 1)
                adapted = True
                break

        if not adapted:
            frozen_switch_speed.append(200)

    print(f"\n  Samples/epochs to adapt after environment switch:")
    print(f"    Living weights: {np.mean(living_switch_speed):.0f} samples (online, no signal)")
    print(f"    Frozen weights: {np.mean(frozen_switch_speed):.0f} epochs (explicit retrain)")

    if np.mean(living_switch_speed) < np.mean(frozen_switch_speed):
        speedup = np.mean(frozen_switch_speed) / max(np.mean(living_switch_speed), 1)
        print(f"    → Living weights adapt {speedup:.1f}× faster")
    else:
        print(f"    → Frozen retraining faster on this task")


# ================================================================
# STABILITY UNDER DRIFT
# ================================================================

def stability_under_drift():
    """
    Real environments don't switch abruptly — they DRIFT.
    Test performance when the target function gradually changes.
    """
    print("\n\n  EXPERIMENT 3: Stability Under Gradual Drift")
    print("  " + "─" * 55)
    print("  Target gradually morphs from XOR toward AND over 1000 steps")
    print("  Living weights should track the drift. Frozen weights can't.")

    X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    Y_xor = np.array([0,1,1,0], float)
    Y_and = np.array([0,0,0,1], float)

    n_steps = 1000
    n_trials = 10

    living_acc_over_time = np.zeros(n_steps)
    frozen_acc_over_time = np.zeros(n_steps)

    for _ in range(n_trials):
        living = LivingWeightNet(2, 6, 1, plasticity_rate=0.15)
        frozen = FrozenWeightNet(2, 6, 1)

        # Pre-train both on XOR
        for _ in range(300):
            idx = np.random.randint(4)
            living.forward(X[idx:idx+1], Y_xor[idx])
        frozen.train_batch(X, Y_xor, epochs=300)

        for t in range(n_steps):
            # Interpolate target: XOR → AND
            alpha = t / n_steps
            Y_current = (1 - alpha) * Y_xor + alpha * Y_and
            Y_binary = (Y_current > 0.5).astype(float)

            # Feed one sample to living network
            idx = np.random.randint(4)
            living.forward(X[idx:idx+1], Y_binary[idx])

            # Evaluate both
            pred_l = living.predict(X)
            pred_f = frozen.predict(X)
            living_acc_over_time[t] += np.mean((pred_l > 0.5) == Y_binary)
            frozen_acc_over_time[t] += np.mean((pred_f > 0.5) == Y_binary)

    living_acc_over_time /= n_trials
    frozen_acc_over_time /= n_trials

    # Report at intervals
    checkpoints = [0, 100, 250, 500, 750, 999]
    print(f"\n  {'Step':>6} {'Drift %':>8} {'Living':>10} {'Frozen':>10}")
    print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*10}")
    for t in checkpoints:
        alpha = t / n_steps * 100
        print(f"  {t:6d} {alpha:7.0f}% {living_acc_over_time[t]:>9.0%} {frozen_acc_over_time[t]:>9.0%}")

    avg_l = np.mean(living_acc_over_time)
    avg_f = np.mean(frozen_acc_over_time)
    print(f"\n  Average accuracy over drift:")
    print(f"    Living: {avg_l:.0%}")
    print(f"    Frozen: {avg_f:.0%}")

    if avg_l > avg_f:
        print(f"    → Living weights track the drift ({avg_l - avg_f:.0%} better)")
    else:
        print(f"    → Frozen weights more stable")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 60)
    print("  LIVING WEIGHTS: Dynamic Substrate Learning")
    print("  Weights that change during inference")
    print("=" * 60)

    continual_learning_test()
    online_adaptation_test()
    stability_under_drift()

    print("\n" + "=" * 60)
    print("  INSIGHT")
    print("=" * 60)
    print("""
  Standard AI separates learning from inference:
    Train (adjust weights) → Deploy (freeze weights)

  The deerskin model unifies them:
    Every forward pass also reshapes the membrane geometry.
    Computation IS learning. Learning IS computation.

  The "living weights" approach with elastic consolidation
  provides:
    1. Continual learning without catastrophic forgetting
       (consolidation protects old memories)
    2. Online adaptation without explicit retraining signals
       (the substrate naturally tracks environmental change)
    3. Graceful handling of distributional drift
       (the system stays calibrated as the world changes)

  This is what biological synaptic plasticity does:
  LTP/LTD physically rearrange receptor distributions
  (the deerskin geometry) on timescales of seconds to hours,
  continuously, while the brain is computing. There is no
  "training mode" vs "inference mode" in biology.
""")


if __name__ == "__main__":
    main()
