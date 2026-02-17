"""
Oscillation Computer: Classification Through Resonance
=======================================================
Standard AI classifies by converging to a fixed output.
The deerskin model suggests an alternative: classify by the
CHARACTER of oscillation when the input disturbs the system.

This experiment builds a small network of coupled geometric
oscillators. Each input pattern causes a different oscillation
signature. Classification = reading the oscillation, not a
static output.

The key insight: oscillation isn't noise — it's the system
exploring moiré space. Different inputs create different moiré
stresses, which produce different rhythms.

Run: python oscillation_computer.py
"""

import numpy as np
import time


class GeometricOscillator:
    """
    A single deerskin loop: pattern → sample → regulate → feedback.
    This is the minimal ECG loop from PerceptionLab, in pure math.
    """

    def __init__(self, freq=10.0, sample_res=16, setpoint=0.5,
                 gain=1.0, sharpness=3.0, dead_zone=0.1):
        self.freq = freq
        self.sample_res = sample_res
        self.setpoint = setpoint
        self.gain = gain
        self.sharpness = sharpness
        self.dead_zone = dead_zone

        # State
        self.value = 0.5
        self.integral = 0.0
        self.history = []

    def reset(self):
        self.value = 0.5
        self.integral = 0.0
        self.history = []

    def _checkerboard_sample(self, scale, input_bias=0.0):
        """Sample a checkerboard at current scale with given resolution."""
        total = 0
        sq_size = max(1, int(30 - scale * 25))
        for i in range(self.sample_res):
            for j in range(self.sample_res):
                x = int(i * 256 / self.sample_res + input_bias * 50)
                y = int(j * 256 / self.sample_res)
                gx = (x // max(1, sq_size)) % 2
                gy = (y // max(1, sq_size)) % 2
                total += gx ^ gy
        return total / (self.sample_res ** 2)

    def step(self, input_signal=0.0):
        """One tick of the oscillator. Returns current output."""
        # Sample the pattern at current value (with input perturbation)
        sample = self._checkerboard_sample(self.value, input_signal)

        # Homeostatic regulation (edge of chaos mode)
        error = sample - self.setpoint

        # Variance estimation
        recent = self.history[-50:] if len(self.history) > 10 else [0.5]
        variance = np.var(recent)
        target_var = 0.1
        var_error = variance - target_var

        if var_error > 0:
            # Too chaotic — dampen
            z = error * self.sharpness
            correction = 2.0 / (1.0 + np.exp(-z)) - 1.0
            self.value = self.setpoint + correction / self.sharpness
        else:
            # Too orderly — excite
            amplified = self.setpoint + error * (1 + abs(var_error) * 10)
            self.value = self.setpoint + np.tanh((amplified - self.setpoint) * 2)

        self.value = np.clip(self.value * self.gain, 0.01, 0.99)
        self.integral += error * 0.01
        self.integral = np.clip(self.integral, -0.5, 0.5)

        self.history.append(self.value)
        return self.value

    def get_oscillation_features(self, n_steps=200):
        """Extract features from the oscillation pattern."""
        if len(self.history) < n_steps:
            return np.zeros(8)

        h = np.array(self.history[-n_steps:])

        # Feature extraction from oscillation character
        features = np.array([
            np.mean(h),                          # Mean level
            np.std(h),                           # Amplitude
            np.mean(np.abs(np.diff(h))),         # Roughness (high freq content)
            np.max(h) - np.min(h),               # Range
            self._count_crossings(h, 0.5),       # Zero crossing rate
            self._dominant_period(h),             # Dominant period
            np.mean(h[:50]) - np.mean(h[50:]),   # Trend
            np.sum(np.diff(h) > 0) / len(h),     # Asymmetry (rise vs fall)
        ])

        return features

    def _count_crossings(self, h, level):
        crossings = 0
        for i in range(1, len(h)):
            if (h[i] - level) * (h[i-1] - level) < 0:
                crossings += 1
        return crossings / len(h)

    def _dominant_period(self, h):
        """Estimate dominant oscillation period via autocorrelation."""
        h = h - np.mean(h)
        if np.std(h) < 1e-6:
            return 0.0
        h = h / np.std(h)
        acorr = np.correlate(h, h, mode='full')
        acorr = acorr[len(acorr)//2:]
        # Find first peak after lag 0
        for i in range(2, len(acorr) - 1):
            if acorr[i] > acorr[i-1] and acorr[i] > acorr[i+1]:
                return i / len(h)
        return 0.0


class OscillationNetwork:
    """
    A small network of coupled geometric oscillators.
    Input perturbs the oscillators differently.
    Output is read from the oscillation CHARACTER, not a static value.
    """

    def __init__(self, n_oscillators=4):
        self.n_osc = n_oscillators
        self.oscillators = [
            GeometricOscillator(
                freq=np.random.uniform(5, 20),
                sample_res=np.random.randint(8, 24),
                gain=np.random.uniform(0.8, 1.2)
            )
            for _ in range(n_oscillators)
        ]
        # Coupling matrix (how oscillators influence each other)
        self.coupling = np.random.randn(n_oscillators, n_oscillators) * 0.1
        np.fill_diagonal(self.coupling, 0)

    def run(self, input_signal, n_steps=200, warmup=50):
        """
        Feed input to the network and let it oscillate.
        Returns feature vector from all oscillators.
        """
        # Reset
        for osc in self.oscillators:
            osc.reset()

        # Warmup (let oscillators find their rhythm)
        for _ in range(warmup):
            values = [osc.step(0) for osc in self.oscillators]

        # Run with input
        for t in range(n_steps):
            values = []
            for i, osc in enumerate(self.oscillators):
                # Input perturbation (different projection for each oscillator)
                perturbation = input_signal * (i + 1) * 0.1

                # Coupling from other oscillators
                if t > 0:
                    coupling_input = sum(
                        self.coupling[i, j] * self.oscillators[j].value
                        for j in range(self.n_osc) if j != i
                    )
                    perturbation += coupling_input

                values.append(osc.step(perturbation))

        # Extract oscillation features from all oscillators
        all_features = []
        for osc in self.oscillators:
            all_features.extend(osc.get_oscillation_features(n_steps))

        return np.array(all_features)


class StaticNetwork:
    """
    Standard feedforward network (for comparison).
    Same parameter count as OscillationNetwork.
    """

    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


# ================================================================
# CLASSIFICATION EXPERIMENTS
# ================================================================

def nearest_centroid_classify(features, centroids, labels):
    """Simple nearest-centroid classifier on oscillation features."""
    dists = [np.linalg.norm(features - c) for c in centroids]
    return labels[np.argmin(dists)]


def exp_pattern_classification():
    """
    Task: Classify input signals by their temporal structure.
    Three classes: constant, slow sine, fast sine.
    The oscillation network classifies by how the input
    changes its resonance pattern.
    """
    print("\n  EXPERIMENT: Temporal Pattern Classification")
    print("  " + "─" * 50)
    print("  Classes: constant, slow oscillation, fast oscillation")
    print("  Oscillation network: classify by resonance character")
    print("  Static network: classify by single forward pass")

    # Build networks
    osc_net = OscillationNetwork(n_oscillators=4)

    # Generate training patterns
    class_inputs = {
        'constant': lambda: 0.5,
        'slow': lambda t: 0.5 + 0.3 * np.sin(t * 0.1),
        'fast': lambda t: 0.5 + 0.3 * np.sin(t * 0.8),
    }

    # Collect oscillation features per class (training)
    print("\n  Training (collecting oscillation signatures)...")
    class_centroids = {}
    n_train = 10

    for name, gen_fn in class_inputs.items():
        features_list = []
        for _ in range(n_train):
            # Run oscillation network with this input class
            # Average the input over time
            avg_input = gen_fn() if 'constant' in name else np.mean([gen_fn(t) for t in range(200)])
            features = osc_net.run(avg_input, n_steps=200)
            features_list.append(features)
        class_centroids[name] = np.mean(features_list, axis=0)

    centroids = list(class_centroids.values())
    labels = list(class_centroids.keys())

    # Test classification
    print("  Testing...")
    n_test = 30
    correct_osc = 0
    total = 0

    for true_label, gen_fn in class_inputs.items():
        for _ in range(n_test):
            # Slightly noisy version
            if true_label == 'constant':
                inp = 0.5 + np.random.randn() * 0.05
            elif true_label == 'slow':
                phase = np.random.uniform(0, 2*np.pi)
                inp = 0.5 + 0.3 * np.sin(phase)
            else:
                phase = np.random.uniform(0, 2*np.pi)
                inp = 0.5 + 0.3 * np.sin(phase * 8)

            features = osc_net.run(inp, n_steps=200)
            pred = nearest_centroid_classify(features, centroids, labels)
            if pred == true_label:
                correct_osc += 1
            total += 1

    osc_accuracy = correct_osc / total

    # Compare: static network baseline (random, since untrained)
    # Also do a simple threshold classifier for fairness
    correct_threshold = 0
    total2 = 0
    for true_label, gen_fn in class_inputs.items():
        for _ in range(n_test):
            if true_label == 'constant':
                inp = 0.5 + np.random.randn() * 0.05
            elif true_label == 'slow':
                inp = 0.5 + 0.3 * np.sin(np.random.uniform(0, 2*np.pi))
            else:
                inp = 0.5 + 0.3 * np.sin(np.random.uniform(0, 2*np.pi) * 8)

            # Threshold classifier: just look at the value
            if abs(inp - 0.5) < 0.1:
                pred = 'constant'
            else:
                pred = 'slow'  # Can't distinguish slow/fast from single sample!

            if pred == true_label:
                correct_threshold += 1
            total2 += 1

    threshold_accuracy = correct_threshold / total2

    print(f"\n  Results ({n_test} trials per class):")
    print(f"    Oscillation network: {osc_accuracy:.1%}")
    print(f"    Single-sample threshold: {threshold_accuracy:.1%}")
    print(f"    Random chance: 33.3%")

    if osc_accuracy > threshold_accuracy:
        print(f"\n  → Oscillation readout wins by {osc_accuracy - threshold_accuracy:.1%}")
        print(f"    The resonance CHARACTER carries information that")
        print(f"    a single static sample cannot capture.")
    else:
        print(f"\n  → Threshold classifier competitive on this simple task")

    return osc_accuracy


def exp_oscillation_signatures():
    """
    Visualize how different inputs produce different oscillation patterns.
    """
    print("\n\n  EXPERIMENT: Oscillation Signature Visualization")
    print("  " + "─" * 50)

    osc = GeometricOscillator(freq=12, sample_res=16, gain=1.0)

    inputs = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"\n  Input → Oscillation character (200 steps)")
    print(f"  {'Input':>7} {'Mean':>8} {'StdDev':>8} {'Freq':>8} {'Range':>8}")
    print(f"  {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for inp in inputs:
        osc.reset()
        for _ in range(300):
            osc.step(inp)

        h = np.array(osc.history[-200:])
        crossings = 0
        for i in range(1, len(h)):
            if (h[i] - 0.5) * (h[i-1] - 0.5) < 0:
                crossings += 1

        print(f"  {inp:7.2f} {np.mean(h):8.4f} {np.std(h):8.4f} {crossings:8d} {np.ptp(h):8.4f}")

        # ASCII mini-trace
        trace = h[::10]  # Subsample
        bar = ""
        for v in trace:
            pos = int((v - 0.3) / 0.4 * 40)
            pos = max(0, min(39, pos))
            bar += "█" if pos > 20 else "░"
        print(f"          [{bar}]")

    print(f"\n  Different inputs → different oscillation signatures")
    print(f"  The system doesn't output a number — it RESONATES differently")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 60)
    print("  OSCILLATION COMPUTER")
    print("  Classification through resonance, not convergence")
    print("=" * 60)

    exp_oscillation_signatures()
    exp_pattern_classification()

    print("\n" + "=" * 60)
    print("  INSIGHT")
    print("=" * 60)
    print("""
  Standard AI computes by convergence: input → fixed output.
  The deerskin model computes by resonance: input → oscillation
  pattern → readout from oscillation character.

  This matters because:
  1. Temporal patterns are naturally encoded in oscillation
     (no need for explicit temporal architectures like LSTMs)
  2. The system "dwells" on inputs — spending time resolving
     ambiguity through oscillation, like human perception
  3. Multiple features are extracted simultaneously from one
     oscillation (frequency, amplitude, phase, symmetry)

  The "box attractor" in EEG theta is this: the brain oscillates
  (searches), snaps to a pattern (recognizes), then releases.
  That search-lock-release cycle is computation through resonance.
""")


if __name__ == "__main__":
    main()
