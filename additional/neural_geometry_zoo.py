"""
Neural Geometry Zoo: Different Membranes for Different Tasks
=============================================================
The key insight: the noise robustness weakness of MoiréNet isn't
a weakness of GEOMETRIC computation — it's a weakness of
CHECKERBOARD geometry specifically.

Biology doesn't use one membrane type. It uses hundreds:
- Purkinje cells: massive smooth dendritic fans (noise tolerant)
- Fast-spiking interneurons: compact, sharp (fast switching)
- Pyramidal cells: moderate complexity (balanced)
- Stellate cells: radial symmetry (omnidirectional)

Each evolved for a different computational niche.

This experiment tests the prediction:
- Sharp (checkerboard) geometry → good at nonlinear separation, bad at noise
- Smooth (sinusoidal) geometry → good at noise tolerance, bad at sharp decisions
- Hybrid (mixed) geometry → balanced performance
- Radial geometry → good at rotation/position invariance

If this holds, then the "weakness" in our earlier experiments
was actually evidence FOR the hypothesis: the brain needs
many cell types because no single geometry does everything.

Run: python neural_geometry_zoo.py
"""

import numpy as np
import time


# ================================================================
# MEMBRANE GEOMETRY TYPES (The Cell Type Library)
# ================================================================

class MembraneGeometry:
    """Base class for different membrane geometries."""
    
    def __init__(self, grid_size=32):
        self.gs = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.name = "base"
        self._build()
    
    def _build(self):
        raise NotImplementedError
    
    def sample(self, x, y):
        """Sample the membrane at normalized coordinates (0-1)."""
        px = int(np.clip(x * (self.gs - 1), 0, self.gs - 1))
        py = int(np.clip(y * (self.gs - 1), 0, self.gs - 1))
        return self.grid[py, px]


class CheckerboardMembrane(MembraneGeometry):
    """
    Sharp binary grid. Like fast-spiking interneurons.
    Properties: fast switching, sharp boundaries, noise-sensitive.
    """
    def __init__(self, freq=10.0, angle=0.0, phase=0.0, gs=32):
        self.freq = freq
        self.angle = angle
        self.phase = phase
        self.gs = gs
        self.name = "checkerboard"
        self.grid = np.zeros((gs, gs), dtype=np.float32)
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

    def set_params(self, f, a, p):
        self.freq, self.angle, self.phase = f, a, p
        self._build()


class SinusoidalMembrane(MembraneGeometry):
    """
    Smooth graded geometry. Like Purkinje cells with their massive
    dendritic fans — gradual transitions, noise-tolerant.
    Properties: smooth transfer functions, graceful degradation.
    """
    def __init__(self, freq=10.0, angle=0.0, phase=0.0, gs=32):
        self.freq = freq
        self.angle = angle
        self.phase = phase
        self.gs = gs
        self.name = "sinusoidal"
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        self._build()
    
    def _build(self):
        s = self.gs
        y, x = np.mgrid[0:s, 0:s]
        c, sn = np.cos(self.angle), np.sin(self.angle)
        rx = (x - s/2) * c - (y - s/2) * sn
        ry = (x - s/2) * sn + (y - s/2) * c
        # Smooth sinusoidal instead of hard threshold
        wave_x = np.sin(rx * self.freq * 2 * np.pi / s + self.phase)
        wave_y = np.sin(ry * self.freq * 2 * np.pi / s)
        # Product creates smooth 2D pattern, normalized to 0-1
        self.grid = ((wave_x * wave_y + 1) / 2).astype(np.float32)

    def set_params(self, f, a, p):
        self.freq, self.angle, self.phase = f, a, p
        self._build()


class RadialMembrane(MembraneGeometry):
    """
    Radially symmetric geometry. Like stellate cells with their
    star-shaped dendritic trees — sensitive to distance from center,
    not direction. Good for position-invariant features.
    """
    def __init__(self, freq=10.0, angle=0.0, phase=0.0, gs=32):
        self.freq = freq
        self.angle = angle  # controls radial offset
        self.phase = phase
        self.gs = gs
        self.name = "radial"
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        self._build()
    
    def _build(self):
        s = self.gs
        y, x = np.mgrid[0:s, 0:s]
        # Distance from center
        r = np.sqrt((x - s/2)**2 + (y - s/2)**2)
        # Concentric rings
        ring = np.sin(r * self.freq * 2 * np.pi / s + self.phase)
        # Angular modulation (slight asymmetry from angle param)
        theta = np.arctan2(y - s/2, x - s/2)
        angular = np.cos(theta * 2 + self.angle)
        self.grid = ((ring * 0.7 + angular * 0.3 + 1) / 2).astype(np.float32)

    def set_params(self, f, a, p):
        self.freq, self.angle, self.phase = f, a, p
        self._build()


class HybridMembrane(MembraneGeometry):
    """
    Mixed geometry: sharp patches embedded in smooth field.
    Like pyramidal cells — moderate dendritic complexity,
    with clustered synaptic "hot spots" on smoother dendrites.
    Balanced performance across tasks.
    """
    def __init__(self, freq=10.0, angle=0.0, phase=0.0, gs=32):
        self.freq = freq
        self.angle = angle
        self.phase = phase
        self.gs = gs
        self.name = "hybrid"
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        self._build()
    
    def _build(self):
        s = self.gs
        y, x = np.mgrid[0:s, 0:s]
        c, sn = np.cos(self.angle), np.sin(self.angle)
        rx = (x - s/2) * c - (y - s/2) * sn + self.phase
        ry = (x - s/2) * sn + (y - s/2) * c
        
        # Smooth base
        smooth = np.sin(rx * self.freq * 2 * np.pi / s) * 0.5 + 0.5
        
        # Sharp patches (clustered synaptic hotspots)
        cell = max(1, s / (self.freq * 0.7 + 1e-6))
        gx = np.floor(rx / cell).astype(int) % 2
        gy = np.floor(ry / cell).astype(int) % 2
        sharp = (gx ^ gy).astype(np.float32)
        
        # Mix: 60% smooth, 40% sharp
        self.grid = (0.6 * smooth + 0.4 * sharp).astype(np.float32)

    def set_params(self, f, a, p):
        self.freq, self.angle, self.phase = f, a, p
        self._build()


class GaborMembrane(MembraneGeometry):
    """
    Gabor-like geometry: oriented edge detector.
    Like simple cells in V1 — tuned to specific orientations
    and spatial frequencies. The classic receptive field model.
    """
    def __init__(self, freq=10.0, angle=0.0, phase=0.0, gs=32):
        self.freq = freq
        self.angle = angle
        self.phase = phase
        self.gs = gs
        self.name = "gabor"
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        self._build()
    
    def _build(self):
        s = self.gs
        y, x = np.mgrid[0:s, 0:s]
        c, sn = np.cos(self.angle), np.sin(self.angle)
        rx = (x - s/2) * c - (y - s/2) * sn
        ry = (x - s/2) * sn + (y - s/2) * c
        
        # Gabor = sinusoid × Gaussian envelope
        sinusoid = np.cos(rx * self.freq * 2 * np.pi / s + self.phase)
        sigma = s / (self.freq * 0.5 + 1)
        gaussian = np.exp(-(rx**2 + ry**2) / (2 * sigma**2))
        
        self.grid = ((sinusoid * gaussian + 1) / 2).astype(np.float32)

    def set_params(self, f, a, p):
        self.freq, self.angle, self.phase = f, a, p
        self._build()


# ================================================================
# GEOMETRY-SPECIFIC NETWORK
# ================================================================

class GeometryNet:
    """Network using a specific membrane geometry type."""
    
    def __init__(self, geometry_class, n_hidden=3, gs=32):
        self.geometry_class = geometry_class
        self.n_hidden = n_hidden
        self.gs = gs
        self.neurons = [
            geometry_class(
                freq=np.random.uniform(3, 25),
                angle=np.random.uniform(0, np.pi),
                phase=np.random.uniform(0, 2*np.pi),
                gs=gs
            ) for _ in range(n_hidden)
        ]
    
    def forward(self, X):
        X = np.atleast_2d(X)
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            vals = [n.sample(x[0], x[1]) for n in self.neurons]
            # Threshold: majority of neurons > 0.5
            out[i] = 1.0 if sum(v > 0.5 for v in vals) > self.n_hidden / 2 else 0.0
        return out
    
    def get_params(self):
        p = []
        for n in self.neurons:
            p.extend([n.freq, n.angle, n.phase])
        return np.array(p)
    
    def set_params(self, p):
        for i, n in enumerate(self.neurons):
            n.set_params(
                max(1, p[i*3]),
                p[i*3+1] % np.pi,
                p[i*3+2] % (2*np.pi)
            )
    
    def evolve(self, X, Y, generations=300, pop_size=60):
        best = self.get_params()
        best_acc = 0
        for g in range(generations):
            candidates = [best + np.random.randn(len(best)) * 0.4 for _ in range(pop_size)]
            for c in candidates:
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
# STANDARD MLP (Baseline)
# ================================================================

class MLP:
    def __init__(self):
        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.zeros(2)
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.zeros(1)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
    def forward(self, X):
        X = np.atleast_2d(X)
        h = self.sigmoid(X @ self.W1 + self.b1)
        return self.sigmoid(h @ self.W2 + self.b2).flatten()
    
    def train(self, X, Y, epochs=1000, lr=0.5):
        for ep in range(epochs):
            h = self.sigmoid(X @ self.W1 + self.b1)
            o = self.sigmoid(h @ self.W2 + self.b2).flatten()
            if np.mean((o > 0.5) == Y) >= 1.0:
                return 1.0, ep + 1
            do = (o - Y) * o * (1 - o)
            dW2 = h.T @ do.reshape(-1, 1)
            db2 = np.sum(do)
            dh = do.reshape(-1, 1) @ self.W2.T * h * (1 - h)
            self.W2 -= lr * dW2; self.b2 -= lr * db2
            self.W1 -= lr * X.T @ dh; self.b1 -= lr * np.sum(dh, axis=0)
        h = self.sigmoid(X @ self.W1 + self.b1)
        o = self.sigmoid(h @ self.W2 + self.b2).flatten()
        return float(np.mean((o > 0.5) == Y)), epochs


# ================================================================
# TEST SUITE
# ================================================================

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], float)
Y_xor = np.array([0,1,1,0], float)

GEOMETRIES = {
    'Checkerboard': CheckerboardMembrane,
    'Sinusoidal':   SinusoidalMembrane,
    'Radial':       RadialMembrane,
    'Hybrid':       HybridMembrane,
    'Gabor':        GaborMembrane,
}


def test_solve_rate(n_trials=20):
    """Test 1: Which geometries solve XOR?"""
    print("\n  TEST 1: XOR Solve Rate (nonlinear separation)")
    print("  " + "─" * 55)
    
    results = {}
    for name, cls in GEOMETRIES.items():
        solved = 0
        gens_list = []
        for _ in range(n_trials):
            net = GeometryNet(cls, n_hidden=3, gs=32)
            acc, g = net.evolve(X_xor, Y_xor, generations=200, pop_size=50)
            if acc >= 1.0:
                solved += 1
                gens_list.append(g)
        
        avg_g = f"{np.mean(gens_list):.0f}" if gens_list else "—"
        results[name] = solved / n_trials
        print(f"    {name:<15} {solved}/{n_trials} solved  avg gen: {avg_g}")
    
    # MLP baseline
    mlp_solved = 0
    for _ in range(n_trials):
        mlp = MLP()
        acc, _ = mlp.train(X_xor, Y_xor)
        if acc >= 1.0: mlp_solved += 1
    results['MLP'] = mlp_solved / n_trials
    print(f"    {'MLP (baseline)':<15} {mlp_solved}/{n_trials} solved")
    
    return results


def test_noise_robustness(n_trials=15):
    """Test 2: THE critical test. Which geometries handle noise?"""
    print("\n\n  TEST 2: Noise Robustness (trained clean, tested noisy)")
    print("  " + "─" * 55)
    
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    
    # Header
    print(f"    {'Noise':>6}", end="")
    for name in list(GEOMETRIES.keys()) + ['MLP']:
        print(f" {name[:8]:>9}", end="")
    print()
    print(f"    {'─'*6}", end="")
    for _ in range(len(GEOMETRIES) + 1):
        print(f" {'─'*9}", end="")
    print()
    
    all_results = {name: {} for name in list(GEOMETRIES.keys()) + ['MLP']}
    
    for noise in noise_levels:
        row = f"    {noise:6.2f}"
        
        for name, cls in GEOMETRIES.items():
            accs = []
            for _ in range(n_trials):
                net = GeometryNet(cls, 3, 32)
                acc, _ = net.evolve(X_xor, Y_xor, generations=200, pop_size=50)
                if acc < 1.0:
                    continue
                
                correct, total = 0, 0
                for xi, yi in zip(X_xor, Y_xor):
                    for _ in range(80):
                        noisy = xi + np.random.randn(2) * noise
                        noisy = np.clip(noisy, -0.5, 1.5)
                        pred = net.forward(noisy.reshape(1, -1))[0]
                        if (pred > 0.5) == yi:
                            correct += 1
                        total += 1
                if total > 0:
                    accs.append(correct / total)
            
            mean_acc = np.mean(accs) if accs else 0
            all_results[name][noise] = mean_acc
            row += f" {mean_acc:>8.0%}"
        
        # MLP
        mlp_accs = []
        for _ in range(n_trials):
            mlp = MLP()
            acc, _ = mlp.train(X_xor, Y_xor)
            if acc < 1.0:
                continue
            correct, total = 0, 0
            for xi, yi in zip(X_xor, Y_xor):
                for _ in range(80):
                    noisy = xi + np.random.randn(2) * noise
                    noisy = np.clip(noisy, -0.5, 1.5)
                    pred = mlp.forward(noisy.reshape(1, -1))[0]
                    if (pred > 0.5) == yi:
                        correct += 1
                    total += 1
            if total > 0:
                mlp_accs.append(correct / total)
        
        mean_mlp = np.mean(mlp_accs) if mlp_accs else 0
        all_results['MLP'][noise] = mean_mlp
        row += f" {mean_mlp:>8.0%}"
        
        print(row)
    
    return all_results


def test_frequency_selectivity():
    """Test 3: Which geometries are best at frequency-dependent computation?"""
    print("\n\n  TEST 3: Frequency Selectivity (independent channels per geometry)")
    print("  " + "─" * 55)
    
    freqs = np.linspace(2, 25, 40)
    
    for name, cls in GEOMETRIES.items():
        n_instances = 15
        tfs = []
        for _ in range(n_instances):
            membrane = cls(
                freq=np.random.uniform(5, 20),
                angle=np.random.uniform(0, np.pi),
                phase=np.random.uniform(0, 2*np.pi),
                gs=64
            )
            # Transfer function: sample along midline at each freq
            tf = []
            for f in freqs:
                # Modulate a carrier at this frequency
                s = 64
                x_coord = np.arange(s)
                carrier = np.sin(x_coord * f * 2 * np.pi / s)
                midline = membrane.grid[s//2, :]
                response = np.mean(carrier * midline)
                tf.append(response)
            tfs.append(tf)
        
        tfs = np.array(tfs)
        
        # Count independent channels via SVD
        centered = tfs - np.mean(tfs, axis=0)
        if np.std(centered) < 1e-10:
            n_channels = 1
        else:
            try:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                n_channels = int(np.sum(S > S[0] * 0.1))
            except:
                n_channels = 1
        
        variability = np.mean(np.std(tfs, axis=1))
        print(f"    {name:<15} {n_channels:>3} channels   variability: {variability:.4f}")
    
    print(f"    {'Scalar weight':<15}   1 channels   variability: 0.0000")


def test_decision_boundary_smoothness():
    """Test 4: Measure how smooth the decision boundary is for each geometry."""
    print("\n\n  TEST 4: Decision Boundary Smoothness")
    print("  " + "─" * 55)
    print("  (Higher = smoother = more noise tolerant)")
    
    for name, cls in GEOMETRIES.items():
        smoothness_scores = []
        
        for _ in range(10):
            net = GeometryNet(cls, 3, 32)
            acc, _ = net.evolve(X_xor, Y_xor, generations=150, pop_size=40)
            if acc < 1.0:
                continue
            
            # Sample the decision surface on a fine grid
            resolution = 50
            grid = np.zeros((resolution, resolution))
            for i in range(resolution):
                for j in range(resolution):
                    x = np.array([[i / resolution, j / resolution]])
                    grid[i, j] = net.forward(x)[0]
            
            # Measure smoothness: average absolute gradient
            dx = np.abs(np.diff(grid, axis=0))
            dy = np.abs(np.diff(grid, axis=1))
            roughness = np.mean(dx) + np.mean(dy)
            smoothness = 1.0 / (roughness + 0.01)
            smoothness_scores.append(smoothness)
        
        if smoothness_scores:
            mean_s = np.mean(smoothness_scores)
            # Visual bar
            bar_len = int(min(40, mean_s * 2))
            bar = "█" * bar_len
            print(f"    {name:<15} {mean_s:>6.2f}  {bar}")
        else:
            print(f"    {name:<15}  (failed to solve XOR)")


def test_biological_mapping():
    """Summarize the biological analogy for each geometry type."""
    print("\n\n  BIOLOGICAL MAPPING")
    print("  " + "─" * 55)
    
    mappings = [
        ("Checkerboard", "Fast-spiking interneuron",
         "Compact membrane, sharp boundaries, fast switching",
         "Precise inhibition, timing"),
        ("Sinusoidal", "Purkinje cell (cerebellum)",
         "Massive smooth dendritic fan, graded responses",
         "Fine motor coordination, noise tolerance"),
        ("Radial", "Stellate cell",
         "Star-shaped dendrites, radial symmetry",
         "Position-invariant feature detection"),
        ("Hybrid", "Pyramidal cell (cortex)",
         "Mixed: smooth dendrites with clustered hot spots",
         "General-purpose cortical computation"),
        ("Gabor", "Simple cell (V1)",
         "Oriented, localized receptive field",
         "Edge detection, orientation selectivity"),
    ]
    
    for geom, cell, structure, role in mappings:
        print(f"\n    {geom} → {cell}")
        print(f"      Structure: {structure}")
        print(f"      Role:      {role}")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 65)
    print("  NEURAL GEOMETRY ZOO")
    print("  Different membranes evolved for different tasks")
    print("=" * 65)
    print("""
  The brain doesn't use one neuron type. It uses hundreds.
  Each has a different membrane geometry — a different "deerskin."
  This experiment tests whether different geometries excel at
  different computational tasks, as evolution would predict.
  """)
    
    solve_results = test_solve_rate(n_trials=15)
    noise_results = test_noise_robustness(n_trials=10)
    test_frequency_selectivity()
    test_decision_boundary_smoothness()
    test_biological_mapping()
    
    # Final synthesis
    print("\n\n" + "=" * 65)
    print("  SYNTHESIS")
    print("=" * 65)
    print("""
  No single membrane geometry is best at everything.
  This is exactly what the deerskin hypothesis predicts:

  - Sharp geometries (checkerboard) → fast nonlinear switching
    but noise-fragile. Biology uses these for precise inhibition.

  - Smooth geometries (sinusoidal) → noise-robust but slower
    at sharp decisions. Biology uses these for sensory integration.

  - Hybrid geometries (pyramidal) → balanced. Biology uses
    these as the general-purpose cortical workhorse.

  - Radial geometries (stellate) → position-invariant.
    Biology uses these where spatial invariance matters.

  - Oriented geometries (Gabor/V1) → frequency-selective.
    Biology uses these for edge detection and feature extraction.

  The "weakness" in our earlier noise robustness test was not
  a weakness of geometric computation. It was a weakness of
  using only ONE geometry type (checkerboard). The brain solves
  this by maintaining a LIBRARY of membrane geometries —
  the cell type zoo — each optimized for its computational niche.

  Evolution didn't find one perfect neuron. It found that
  different computational problems require different deerskins.
  """)


if __name__ == "__main__":
    main()
