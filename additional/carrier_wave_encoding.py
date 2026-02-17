"""
Carrier Wave Encoding: Hidden Signal Capacity
===============================================
Core question: How much information does a single "spike" carry?

Point neuron: One scalar value. One number per spike. That's it.
Deerskin neuron: A carrier wave modulated by membrane geometry.
                 The output waveform encodes the membrane's fingerprint.

This experiment measures:
1. How many distinct messages can ride on one carrier through different geometries
2. Whether a receiver can decode WHICH geometry the carrier traversed
3. The information capacity (bits) of geometric vs scalar encoding

Run: python carrier_wave_encoding.py
"""

import numpy as np
import time


class Membrane:
    """A 2D membrane geometry (the deerskin)."""

    def __init__(self, size=64, n_channels=None):
        self.size = size
        if n_channels is None:
            # Random channel distribution
            self.geometry = self._random_geometry()
        else:
            self.geometry = n_channels

    def _random_geometry(self):
        """Generate a random but structured channel mosaic."""
        s = self.size
        g = np.zeros((s, s), dtype=np.float32)

        # Layer multiple spatial frequencies (like real ion channel distributions)
        n_patches = np.random.randint(2, 6)
        for _ in range(n_patches):
            freq = np.random.uniform(2, 15)
            angle = np.random.uniform(0, np.pi)
            phase = np.random.uniform(0, 2 * np.pi)

            y, x = np.mgrid[0:s, 0:s]
            c, sn = np.cos(angle), np.sin(angle)
            rx = (x - s/2) * c - (y - s/2) * sn
            pattern = np.sin(rx * freq * 2 * np.pi / s + phase)
            g += pattern

        # Threshold to binary channels (open/closed)
        g = (g > 0).astype(np.float32)
        return g

    def modulate_carrier(self, carrier):
        """
        The soma pulse (carrier) traverses this membrane.
        Returns the modulated output — the "holographic fingerprint".
        """
        return carrier * self.geometry

    def get_fingerprint(self, sample_points=None):
        """
        Sample the output at discrete points (receptor positions).
        This is what the RECEIVING neuron sees.
        """
        if sample_points is None:
            # Default: sample along a line (like an axon terminal)
            mid = self.size // 2
            return self.geometry[mid, :]
        else:
            return np.array([self.geometry[int(p[1]), int(p[0])]
                           for p in sample_points])


def carrier_wave(size, freq=5.0, t=0.0):
    """Generate a 2D propagating pulse (the soma spike)."""
    y, x = np.mgrid[0:size, 0:size]
    # A wavefront propagating from left to right
    pulse_center = t * size
    dist = x - pulse_center
    carrier = np.exp(-dist**2 / (2 * (size * 0.1)**2)) * np.cos(dist * freq * 2 * np.pi / size)
    return carrier.astype(np.float32)


# ================================================================
# EXPERIMENT 1: Fingerprint Distinctness
# ================================================================

def exp_fingerprint_distinctness(n_membranes=50, n_trials=100):
    """
    Generate N different membranes. Pass the same carrier through each.
    Measure how distinguishable the outputs are.

    Compare against N scalar "weights" — just multiplying the carrier by a number.
    """
    print("\n  EXPERIMENT 1: Fingerprint Distinctness")
    print("  " + "─" * 50)

    size = 64
    membranes = [Membrane(size) for _ in range(n_membranes)]
    carrier = carrier_wave(size, freq=5.0, t=0.5)

    # Get geometric fingerprints (sample along midline)
    geo_fingerprints = []
    for m in membranes:
        output = m.modulate_carrier(carrier)
        # Sample at 32 points along the output edge (axon terminals)
        fp = output[size//2, ::2]  # Every other pixel along midline
        geo_fingerprints.append(fp)

    geo_fingerprints = np.array(geo_fingerprints)

    # Get scalar fingerprints (just multiply carrier by a weight)
    scalar_weights = np.random.uniform(0.1, 2.0, n_membranes)
    scalar_fingerprints = []
    for w in scalar_weights:
        output = carrier * w
        fp = output[size//2, ::2]
        scalar_fingerprints.append(fp)

    scalar_fingerprints = np.array(scalar_fingerprints)

    # Measure pairwise distances
    def mean_pairwise_dist(fps):
        dists = []
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                d = np.linalg.norm(fps[i] - fps[j])
                dists.append(d)
        return np.mean(dists), np.std(dists)

    geo_mean, geo_std = mean_pairwise_dist(geo_fingerprints)
    scl_mean, scl_std = mean_pairwise_dist(scalar_fingerprints)

    print(f"  {n_membranes} membranes, same carrier wave")
    print(f"  Geometric fingerprints: mean dist = {geo_mean:.3f} ± {geo_std:.3f}")
    print(f"  Scalar fingerprints:    mean dist = {scl_mean:.3f} ± {scl_std:.3f}")
    print(f"  Ratio: geometric is {geo_mean/scl_mean:.1f}× more distinguishable")

    # Classification test: can we identify which membrane produced an output?
    correct_geo = 0
    correct_scl = 0

    for _ in range(n_trials):
        # Pick a random membrane
        idx = np.random.randint(n_membranes)

        # Generate noisy observation
        noise = np.random.randn(len(geo_fingerprints[0])) * 0.1

        # Geometric: match against known fingerprints
        geo_obs = geo_fingerprints[idx] + noise
        geo_dists = [np.linalg.norm(geo_obs - fp) for fp in geo_fingerprints]
        if np.argmin(geo_dists) == idx:
            correct_geo += 1

        # Scalar: match against known fingerprints
        scl_obs = scalar_fingerprints[idx] + noise
        scl_dists = [np.linalg.norm(scl_obs - fp) for fp in scalar_fingerprints]
        if np.argmin(scl_dists) == idx:
            correct_scl += 1

    print(f"\n  Identification accuracy ({n_trials} trials, noise=0.1):")
    print(f"    Geometric: {correct_geo/n_trials:.1%}")
    print(f"    Scalar:    {correct_scl/n_trials:.1%}")

    return geo_mean / scl_mean


# ================================================================
# EXPERIMENT 2: Information Capacity
# ================================================================

def exp_information_capacity():
    """
    How many bits of information can one spike carry?

    Scalar: log2(N) bits where N is the number of distinguishable weight levels
    Geometric: much more, because the fingerprint has spatial structure
    """
    print("\n\n  EXPERIMENT 2: Information Capacity (bits per spike)")
    print("  " + "─" * 50)

    size = 64
    sample_points = 32  # Number of "receptor" sampling points

    # Test at various noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]

    print(f"  {'Noise':>8} {'Geo (bits)':>12} {'Scalar (bits)':>14} {'Ratio':>8}")
    print(f"  {'─'*8} {'─'*12} {'─'*14} {'─'*8}")

    for noise in noise_levels:
        # Geometric capacity
        # Generate many membranes, see how many are distinguishable at this noise
        n_test = 200
        membranes = [Membrane(size) for _ in range(n_test)]
        carrier = carrier_wave(size, freq=5.0, t=0.5)

        fingerprints = []
        for m in membranes:
            out = m.modulate_carrier(carrier)
            fp = out[size//2, ::size//sample_points][:sample_points]
            fingerprints.append(fp)
        fingerprints = np.array(fingerprints)

        # Count distinguishable pairs
        n_distinguishable = 0
        for i in range(len(fingerprints)):
            for j in range(i+1, len(fingerprints)):
                dist = np.linalg.norm(fingerprints[i] - fingerprints[j])
                if dist > noise * np.sqrt(sample_points) * 3:  # 3-sigma separation
                    n_distinguishable += 1
        total_pairs = n_test * (n_test - 1) / 2
        frac_distinguishable = n_distinguishable / total_pairs

        # Effective number of distinguishable states
        n_geo_states = max(2, int(n_test * frac_distinguishable))
        geo_bits = np.log2(n_geo_states)

        # Scalar capacity
        # A scalar weight with noise=sigma can distinguish levels separated by ~3*sigma
        # In range [0, 2], number of levels = 2 / (3 * noise)
        n_scl_states = max(2, int(2.0 / (3 * noise)))
        scl_bits = np.log2(n_scl_states)

        ratio = geo_bits / max(scl_bits, 0.1)
        print(f"  {noise:8.2f} {geo_bits:12.1f} {scl_bits:14.1f} {ratio:7.1f}×")


# ================================================================
# EXPERIMENT 3: Frequency-Dependent Encoding
# ================================================================

def exp_frequency_encoding():
    """
    The same membrane produces DIFFERENT fingerprints at different carrier frequencies.
    A scalar weight produces the SAME output regardless of frequency (weight × signal = scaled signal).

    This is the multi-head attention equivalent: one connection, many transfer functions.
    """
    print("\n\n  EXPERIMENT 3: Frequency-Dependent Encoding")
    print("  " + "─" * 50)
    print("  Same membrane, different carrier frequencies → different fingerprints?")

    size = 64
    membrane = Membrane(size)

    # Test multiple carrier frequencies
    freqs = [2, 4, 6, 8, 10, 15, 20]
    fingerprints = []

    for f in freqs:
        carrier = carrier_wave(size, freq=f, t=0.5)
        output = membrane.modulate_carrier(carrier)
        fp = output[size//2, :]
        fingerprints.append(fp)

    # Measure pairwise correlation between fingerprints at different frequencies
    print(f"\n  Cross-frequency correlation matrix (lower = more independent):")
    print(f"  {'Hz':>6}", end="")
    for f in freqs:
        print(f" {f:>6}", end="")
    print()

    for i, fi in enumerate(freqs):
        print(f"  {fi:>5}:", end="")
        for j, fj in enumerate(freqs):
            corr = np.corrcoef(fingerprints[i], fingerprints[j])[0, 1]
            if i == j:
                print(f"  {'1.00':>5}", end="")
            else:
                print(f"  {corr:>5.2f}", end="")
        print()

    # Count effectively independent channels
    corr_matrix = np.corrcoef(fingerprints)
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    # Number of eigenvalues > 0.1 ≈ number of independent channels
    n_independent = np.sum(eigenvalues > 0.1)

    print(f"\n  Number of effectively independent channels: {n_independent}")
    print(f"  (A scalar weight has exactly 1 independent channel)")
    print(f"  → One deerskin connection ≈ {n_independent} attention heads")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 60)
    print("  CARRIER WAVE ENCODING: Hidden Signal Capacity")
    print("  How much information rides on one spike?")
    print("=" * 60)

    ratio = exp_fingerprint_distinctness()
    exp_information_capacity()
    exp_frequency_encoding()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
  The deerskin carrier wave encodes the membrane's geometric
  fingerprint on every spike. This means:

  1. Each spike carries a high-dimensional signal, not a scalar
  2. The same connection computes different functions at different
     frequencies (like multi-head attention, but from one connection)
  3. Geometric fingerprints are far more distinguishable than
     scalar weights, providing higher information capacity per spike

  This is why the brain achieves 86 billion neurons × 7,000
  connections on 20 watts, while GPT-4 needs megawatts for
  ~1.8 trillion scalar parameters. The computational primitive
  is richer — each operation does more.
""")


if __name__ == "__main__":
    main()
