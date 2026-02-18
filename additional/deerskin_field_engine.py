"""
DeerskinEngine: Spatiotemporal Field Anomaly Detection
=======================================================

The 1D anomaly detector failed honestly — 1D scalars don't have
enough spatial structure for geometric computation to exploit.
A membrane can't "see" patterns in a number.

But a 16-channel sensor grid IS a 2D field. Each channel is a
spatial position. The signal IS already a membrane. The moiré
between the incoming spatial pattern and the detector's membrane
geometry computes anomaly features that scalar approaches can't:
spatial coherence, wavefront direction, frequency-space anomalies.

This is what the deerskin was built for.

Applications:
- EEG electrode grids (16-256 channels, each a scalp position)
- Industrial sensor arrays (vibration, temperature, pressure)
- Seismic sensor networks
- Antenna arrays

Run: python deerskin_field_engine.py
Requirements: numpy only
"""

import numpy as np
import time
import sys


# ================================================================
# SPATIOTEMPORAL FIELD GENERATOR
# ================================================================

class FieldGenerator:
    """
    Generates realistic multi-channel spatiotemporal signals.
    Models a 4×4 grid of sensors measuring a wave field.
    """

    def __init__(self, grid_size=4, base_freq=0.03, noise=0.08):
        self.gs = grid_size
        self.n_channels = grid_size * grid_size
        self.base_freq = base_freq
        self.noise = noise

        # Sensor positions
        self.positions = np.array([(i, j) for i in range(grid_size)
                                   for j in range(grid_size)], dtype=float)
        self.positions /= (grid_size - 1)  # Normalize to 0-1

    def normal_field(self, n_samples):
        """
        Generate normal spatiotemporal field.
        Two wavefronts propagating across the grid + noise.
        Each sensor sees the wave at its spatial position.
        """
        t = np.arange(n_samples, dtype=float)
        data = np.zeros((n_samples, self.n_channels))

        # Wave 1: propagating left-to-right
        for ch, (px, py) in enumerate(self.positions):
            phase_delay = px * 2.0  # Spatial phase from position
            data[:, ch] += np.sin(2*np.pi*self.base_freq*t + phase_delay)

        # Wave 2: propagating top-to-bottom, different frequency
        for ch, (px, py) in enumerate(self.positions):
            phase_delay = py * 1.5
            data[:, ch] += 0.5 * np.sin(2*np.pi*self.base_freq*2.3*t + phase_delay)

        # Correlated noise (spatial smoothness)
        for i in range(n_samples):
            raw_noise = np.random.randn(self.n_channels)
            # Smooth spatially
            smooth = np.zeros(self.n_channels)
            for ch in range(self.n_channels):
                weights = np.exp(-np.sum((self.positions - self.positions[ch])**2, axis=1) / 0.3)
                weights /= weights.sum()
                smooth[ch] = weights @ raw_noise
            data[i] += self.noise * smooth

        return data

    def inject_spatial_anomaly(self, data, n_anomalies=15):
        """
        Inject anomalies that have SPATIAL structure:
        - Localized hotspots (one region spikes while others are normal)
        - Wavefront disruptions (propagation pattern breaks)
        - Coherence loss (correlated channels become uncorrelated)
        """
        n, nc = data.shape
        labels = np.zeros(n, dtype=bool)

        for _ in range(n_anomalies):
            anom_type = np.random.choice(['hotspot', 'wavefront', 'coherence'])
            loc = np.random.randint(100, n - 20)

            if anom_type == 'hotspot':
                # Localized spatial spike: one quadrant spikes
                center = self.positions[np.random.randint(nc)]
                dists = np.sqrt(np.sum((self.positions - center)**2, axis=1))
                affected = dists < 0.4
                magnitude = np.random.uniform(3, 6) * np.random.choice([-1, 1])
                duration = np.random.randint(1, 4)
                for dt in range(duration):
                    if loc + dt < n:
                        data[loc + dt, affected] += magnitude * np.exp(-dists[affected] / 0.2)
                        labels[loc + dt] = True

            elif anom_type == 'wavefront':
                # Wavefront breaks: normal propagation pattern disrupted
                duration = np.random.randint(5, 15)
                for dt in range(duration):
                    if loc + dt < n:
                        # Reverse the phase relationship
                        for ch, (px, py) in enumerate(self.positions):
                            data[loc + dt, ch] += 2.0 * np.sin(-px * 3 + py * 2)
                        labels[loc + dt] = True

            elif anom_type == 'coherence':
                # Coherence loss: channels that should be correlated become independent
                duration = np.random.randint(8, 20)
                for dt in range(duration):
                    if loc + dt < n:
                        data[loc + dt] += np.random.randn(nc) * 1.5
                        labels[loc + dt] = True

        return data, labels


# ================================================================
# DEERSKIN FIELD ENGINE
# ================================================================

class DeerskinFieldEngine:
    """
    Multi-channel spatiotemporal anomaly detector using geometric computation.

    The key difference from the 1D engine: the input IS a 2D field.
    No folding needed. Each time step provides a spatial snapshot
    that maps directly onto the membrane grids.

    Architecture:
    - 4 membrane cells (different types, different geometries)
    - Each cell computes moiré between the spatial field and its geometry
    - Living weights adapt the geometry to track normal patterns
    - Anomaly = unexpected moiré pattern (high surprise across cells)
    """

    def __init__(self, grid_size=4, window_size=16, plasticity=0.008):
        self.gs = grid_size
        self.n_channels = grid_size * grid_size
        self.window_size = window_size
        self.plasticity = plasticity

        # Mixed geometry ensemble (different cell types for different features)
        self.cells = []
        cell_configs = [
            # (type, freq, angle) — each detects different spatial patterns
            ('checker', 2.0, 0.0),      # Detects grid-aligned anomalies
            ('checker', 3.0, 0.785),     # Detects diagonal anomalies
            ('sinusoidal', 2.5, 0.0),    # Smooth spatial gradients
            ('sinusoidal', 4.0, 1.57),   # High-freq smooth patterns
            ('radial', 3.0, 0.0),        # Radial patterns from center
            ('gabor', 2.0, 0.0),         # Oriented edge-like features
            ('gabor', 2.0, 1.57),        # Perpendicular orientation
            ('radial', 5.0, 0.0),        # Fine radial patterns
        ]
        for ctype, freq, angle in cell_configs:
            self.cells.append(SpatialCell(grid_size, ctype, freq, angle, plasticity))

        self.n_cells = len(self.cells)
        self.buffer = []
        self.step = 0
        self.warmup = 40

        # Global normalization
        self.running_mean = 0.0
        self.running_std = 1.0

        # Joint feature statistics (across all cells)
        self.feat_mean = None
        self.feat_var = None
        self.score_history = []

        # Parameter count: each cell has 3 geometric params
        self.n_params = self.n_cells * 3

    def process(self, spatial_frame):
        """
        Process one spatial snapshot (grid_size × grid_size values).
        Returns anomaly score in [0, 1].
        """
        frame = np.array(spatial_frame).reshape(self.gs, self.gs)
        self.step += 1

        # Global running normalization (stabilizes over time)
        flat = frame.flatten()
        self.running_mean = 0.99 * self.running_mean + 0.01 * np.mean(flat)
        self.running_std = 0.99 * self.running_std + 0.01 * (np.std(flat) + 1e-8)
        normed = (frame - self.running_mean) / self.running_std

        self.buffer.append(normed.copy())

        # Need warmup for stable statistics
        if self.step < self.warmup:
            return 0.5

        # Temporal context
        buf_len = min(self.window_size, len(self.buffer))
        recent = np.mean(self.buffer[-buf_len:], axis=0)
        temporal_delta = normed - recent

        # Each cell computes its moiré features
        all_features = []
        for cell in self.cells:
            spatial_features = cell.compute_moire(normed)
            temporal_features = cell.compute_moire(temporal_delta)
            features = np.concatenate([spatial_features, temporal_features])
            all_features.append(features)

        # Stack all cell features into one vector
        feat_vec = np.concatenate(all_features)

        # Update feature statistics with robust EMA
        if self.feat_mean is None:
            self.feat_mean = feat_vec.copy()
            self.feat_var = np.ones_like(feat_vec) * 0.01
            return 0.5

        # Mahalanobis-like anomaly score across ALL cells jointly
        diff = feat_vec - self.feat_mean
        z_scores = diff**2 / (self.feat_var + 1e-8)
        raw_score = np.mean(z_scores)

        # Adapt feature statistics (slow, stable)
        alpha = 0.02
        self.feat_mean = (1 - alpha) * self.feat_mean + alpha * feat_vec
        residuals = (feat_vec - self.feat_mean)**2
        self.feat_var = (1 - alpha) * self.feat_var + alpha * residuals

        # Calibrate raw score to [0,1] using score statistics
        self.score_history.append(raw_score)
        if len(self.score_history) > 200:
            self.score_history = self.score_history[-200:]
        score_mean = np.mean(self.score_history)
        score_std = np.std(self.score_history) + 1e-8
        z = (raw_score - score_mean) / score_std
        score = 1.0 / (1.0 + np.exp(-z + 1.0))

        # Living weight adaptation (only during non-anomalous periods)
        if raw_score < score_mean + 2 * score_std:
            for cell_feats, cell in zip(all_features, self.cells):
                surprise = np.mean((cell_feats - np.mean(cell_feats))**2)
                cell.adapt(surprise)

        # Trim buffer
        if len(self.buffer) > self.window_size * 2:
            self.buffer = self.buffer[-self.window_size:]

        return float(score)


class SpatialCell:
    """
    A single membrane geometry that computes moiré with spatial fields.
    No artificial folding — the input grid maps directly onto the membrane.
    """

    def __init__(self, grid_size, cell_type, freq, angle, plasticity):
        self.gs = grid_size
        self.cell_type = cell_type
        self.freq = freq
        self.angle = angle
        self.phase = np.random.uniform(0, 2*np.pi)
        self.plasticity = plasticity

        # Slow-moving consolidated parameters
        self.slow_freq = freq
        self.slow_angle = angle

        # Running statistics for surprise
        self.feat_ema = None
        self.feat_var = None
        self.alpha = 0.05

        self._build_grid()

    def _build_grid(self):
        s = self.gs
        y, x = np.mgrid[0:s, 0:s].astype(float)
        # Normalize to [-1, 1]
        xn = 2 * x / (s - 1) - 1
        yn = 2 * y / (s - 1) - 1
        c, sn = np.cos(self.angle), np.sin(self.angle)
        rx = xn * c - yn * sn
        ry = xn * sn + yn * c

        if self.cell_type == 'checker':
            pattern = np.sign(np.sin(rx * self.freq * np.pi + self.phase) *
                            np.sin(ry * self.freq * np.pi))
            self.grid = (pattern + 1) / 2
        elif self.cell_type == 'sinusoidal':
            wx = np.sin(rx * self.freq * np.pi + self.phase)
            wy = np.sin(ry * self.freq * np.pi)
            self.grid = (wx * wy + 1) / 2
        elif self.cell_type == 'radial':
            r = np.sqrt(xn**2 + yn**2)
            theta = np.arctan2(yn, xn)
            self.grid = (np.sin(r * self.freq * np.pi + self.phase) *
                        np.cos(theta * 2 + self.angle) + 1) / 2
        elif self.cell_type == 'gabor':
            sinusoid = np.cos(rx * self.freq * np.pi + self.phase)
            sigma = 1.0 / (self.freq * 0.3 + 0.5)
            gauss = np.exp(-(rx**2 + ry**2) / (2 * sigma**2))
            self.grid = (sinusoid * gauss + 1) / 2

    def compute_moire(self, field):
        """
        THE CORE OPERATION: moiré between spatial field and membrane geometry.
        This is where geometric computation actually earns its keep —
        the field is REAL 2D data, not folded scalars.
        """
        moire = field * self.grid

        # Extract spatial features from the moiré pattern
        features = np.array([
            np.mean(moire),                                    # Overall activation
            np.std(moire),                                     # Activation spread
            np.mean(moire[:self.gs//2]) - np.mean(moire[self.gs//2:]),  # Top-bottom asymmetry
            np.mean(moire[:, :self.gs//2]) - np.mean(moire[:, self.gs//2:]),  # Left-right asymmetry
            np.mean(np.abs(np.diff(moire, axis=0))),          # Vertical gradient energy
            np.mean(np.abs(np.diff(moire, axis=1))),          # Horizontal gradient energy
            np.sum(moire * self.grid) / (np.sum(self.grid**2) + 1e-8),  # Correlation with membrane
            np.mean((moire - np.mean(moire))**2),             # Variance of moiré
        ])
        return features

    def compute_surprise(self, features):
        if self.feat_ema is None:
            self.feat_ema = features.copy()
            self.feat_var = np.ones_like(features) * 0.01
            return 0.0

        # Mahalanobis-like distance
        diff = features - self.feat_ema
        z_scores = diff**2 / (self.feat_var + 1e-8)
        surprise = np.mean(z_scores)

        # Update running stats
        self.feat_ema = (1 - self.alpha) * self.feat_ema + self.alpha * features
        residuals = (features - self.feat_ema)**2
        self.feat_var = (1 - self.alpha) * self.feat_var + self.alpha * residuals

        return surprise

    def adapt(self, surprise):
        """Living weight adaptation: geometry shifts to track normal patterns."""
        if self.plasticity <= 0:
            return

        # Only adapt when surprise is moderate (not during anomalies)
        gate = np.exp(-(surprise - 0.5)**2 / 2.0)
        lr = gate * self.plasticity

        # Pull toward consolidated memory (elastic consolidation)
        self.freq += (self.slow_freq - self.freq) * lr * 0.2
        self.angle += (self.slow_angle - self.angle) * lr * 0.1
        self.phase += np.random.randn() * lr * 0.15

        self.freq = np.clip(self.freq, 1, 8)

        # Consolidate
        cr = 0.003
        self.slow_freq = (1 - cr) * self.slow_freq + cr * self.freq
        self.slow_angle = (1 - cr) * self.slow_angle + cr * self.angle

        self._build_grid()


# ================================================================
# BASELINE DETECTORS (for fair comparison)
# ================================================================

class ChannelwiseZScore:
    """Z-score per channel, aggregated. Standard approach."""
    def __init__(self, grid_size=4, window=50):
        self.nc = grid_size * grid_size
        self.window = window
        self.buffers = [[] for _ in range(self.nc)]
        self.n_params = 1

    def process(self, frame):
        frame = np.array(frame).flatten()
        zscores = []
        for ch in range(self.nc):
            self.buffers[ch].append(frame[ch])
            if len(self.buffers[ch]) < 5:
                zscores.append(0)
                continue
            recent = self.buffers[ch][-self.window:]
            z = abs(frame[ch] - np.mean(recent)) / (np.std(recent) + 1e-8)
            zscores.append(z)
        max_z = np.max(zscores)
        mean_z = np.mean(zscores)
        # High score if ANY channel is anomalous
        combined = 0.6 * max_z + 0.4 * mean_z
        return 1 / (1 + np.exp(-combined + 2))


class PCADetector:
    """
    PCA-based detector. Learns spatial covariance,
    flags deviations from normal subspace.
    """
    def __init__(self, grid_size=4, window=100, n_components=4):
        self.nc = grid_size * grid_size
        self.window = window
        self.k = n_components
        self.buffer = []
        self.components = None
        self.mean = None
        self.n_params = self.nc * n_components + self.nc  # components + mean

    def process(self, frame):
        frame = np.array(frame).flatten()
        self.buffer.append(frame)

        if len(self.buffer) < 30:
            return 0.5

        # Refit periodically
        if len(self.buffer) % 50 == 0 or self.components is None:
            data = np.array(self.buffer[-self.window:])
            self.mean = np.mean(data, axis=0)
            centered = data - self.mean
            cov = centered.T @ centered / len(centered)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            self.components = eigvecs[:, idx[:self.k]]

        centered = frame - self.mean
        projected = centered @ self.components
        reconstructed = projected @ self.components.T
        error = np.sum((centered - reconstructed)**2)

        return 1 / (1 + np.exp(-error * 0.5 + 2))


class FlattenedMLP:
    """
    MLP autoencoder on flattened spatial input.
    Online learning version — fairest comparison to living weights.
    """
    def __init__(self, grid_size=4, hidden=12, lr=0.01):
        self.nc = grid_size * grid_size
        self.h = hidden
        self.lr = lr
        self.W1 = np.random.randn(self.nc, hidden) * 0.3
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, self.nc) * 0.3
        self.b2 = np.zeros(self.nc)
        self.mean = np.zeros(self.nc)
        self.std = np.ones(self.nc)
        self.count = 0
        self.buffer = []
        self.n_params = self.nc * hidden * 2 + hidden + self.nc

    def process(self, frame):
        frame = np.array(frame).flatten()
        self.buffer.append(frame)
        self.count += 1

        # Update running stats
        self.mean += (frame - self.mean) / self.count
        if self.count > 5:
            self.std = np.std(self.buffer[-100:], axis=0) + 1e-8

        if len(self.buffer) < 10:
            return 0.5

        x = ((frame - self.mean) / self.std).reshape(1, -1)
        h = np.tanh(x @ self.W1 + self.b1)
        r = h @ self.W2 + self.b2
        err_vec = r - x
        error = np.mean(err_vec**2)

        # Online update
        self.W2 -= self.lr * (h.T @ err_vec)
        self.b2 -= self.lr * err_vec.flatten()
        dh = err_vec @ self.W2.T * (1 - h**2)
        self.W1 -= self.lr * (x.T @ dh)
        self.b1 -= self.lr * dh.flatten()

        return float(1 / (1 + np.exp(-error * 3 + 1.5)))


class FrozenMLP:
    """MLP autoencoder, trained on initial window then frozen."""
    def __init__(self, grid_size=4, hidden=12, train_window=200, epochs=50, lr=0.05):
        self.nc = grid_size * grid_size
        self.h = hidden
        self.buffer = []
        self.trained = False
        self.train_window = train_window
        self.epochs = epochs
        self.lr = lr
        self.W1 = np.random.randn(self.nc, hidden) * 0.3
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, self.nc) * 0.3
        self.b2 = np.zeros(self.nc)
        self.mean = None
        self.std = None
        self.n_params = self.nc * hidden * 2 + hidden + self.nc

    def _train(self):
        data = np.array(self.buffer[:self.train_window])
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8
        X = (data - self.mean) / self.std

        for _ in range(self.epochs):
            h = np.tanh(X @ self.W1 + self.b1)
            r = h @ self.W2 + self.b2
            err = r - X
            self.W2 -= self.lr * (h.T @ err) / len(X)
            self.b2 -= self.lr * np.mean(err, axis=0)
            dh = err @ self.W2.T * (1 - h**2)
            self.W1 -= self.lr * (X.T @ dh) / len(X)
            self.b1 -= self.lr * np.mean(dh, axis=0)

        self.trained = True

    def process(self, frame):
        frame = np.array(frame).flatten()
        self.buffer.append(frame)

        if len(self.buffer) < self.train_window:
            return 0.5
        if not self.trained:
            self._train()

        x = ((frame - self.mean) / self.std).reshape(1, -1)
        h = np.tanh(x @ self.W1 + self.b1)
        r = h @ self.W2 + self.b2
        error = np.mean((r - x)**2)
        return float(1 / (1 + np.exp(-error * 3 + 1.5)))


class SpatialCoherenceDetector:
    """
    Measures spatial coherence between neighboring channels.
    Anomalies that disrupt spatial patterns show as coherence drops.
    """
    def __init__(self, grid_size=4, window=30):
        self.gs = grid_size
        self.nc = grid_size * grid_size
        self.window = window
        self.buffer = []
        self.coherence_ema = None
        self.coherence_var = None
        self.alpha = 0.05
        self.n_params = 2  # alpha and window

    def _neighbors(self, idx):
        r, c = idx // self.gs, idx % self.gs
        nbrs = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc_ = r + dr, c + dc
            if 0 <= nr < self.gs and 0 <= nc_ < self.gs:
                nbrs.append(nr * self.gs + nc_)
        return nbrs

    def process(self, frame):
        frame = np.array(frame).flatten()
        self.buffer.append(frame)

        if len(self.buffer) < 5:
            return 0.5

        # Compute spatial coherence: correlation between each channel and its neighbors
        coherences = []
        for ch in range(self.nc):
            nbrs = self._neighbors(ch)
            if not nbrs:
                continue
            nbr_vals = [frame[n] for n in nbrs]
            coherences.append(abs(frame[ch] - np.mean(nbr_vals)))

        coherence = np.mean(coherences)

        if self.coherence_ema is None:
            self.coherence_ema = coherence
            self.coherence_var = 0.01
            return 0.5

        z = abs(coherence - self.coherence_ema) / (np.sqrt(self.coherence_var) + 1e-8)

        self.coherence_ema = (1 - self.alpha) * self.coherence_ema + self.alpha * coherence
        res = (coherence - self.coherence_ema)**2
        self.coherence_var = (1 - self.alpha) * self.coherence_var + self.alpha * res

        return 1 / (1 + np.exp(-z + 1.5))


# ================================================================
# SCENARIOS
# ================================================================

def scenario_hotspot(n=1200):
    """Localized spatial anomalies — one region spikes."""
    fg = FieldGenerator(4, 0.03, 0.08)
    data = fg.normal_field(n)
    data, labels = fg.inject_spatial_anomaly(data, n_anomalies=18)
    return data, labels, "Spatial Hotspot"


def scenario_wavefront_break(n=1200):
    """Normal wave propagation breaks — direction or speed changes."""
    fg = FieldGenerator(4, 0.03, 0.06)
    data = fg.normal_field(n)
    labels = np.zeros(n, dtype=bool)

    # Inject wavefront reversals
    for _ in range(8):
        loc = np.random.randint(150, n - 30)
        duration = np.random.randint(8, 20)
        for dt in range(duration):
            if loc + dt < n:
                for ch, (px, py) in enumerate(fg.positions):
                    # Reverse wave direction
                    data[loc + dt, ch] += 2.5 * np.sin(-(px * 3) + py * 2)
                labels[loc + dt] = True

    return data, labels, "Wavefront Break"


def scenario_coherence_loss(n=1200):
    """
    Spatial coherence breaks — channels that should track together
    become independent. Spatial detectors should catch this.
    Scalar detectors that process channels independently will miss it.
    """
    fg = FieldGenerator(4, 0.03, 0.06)
    data = fg.normal_field(n)
    labels = np.zeros(n, dtype=bool)

    for _ in range(10):
        loc = np.random.randint(150, n - 30)
        duration = np.random.randint(10, 25)
        # Random subset of channels lose coherence
        n_affected = np.random.randint(4, 12)
        affected = np.random.choice(fg.n_channels, n_affected, replace=False)
        for dt in range(duration):
            if loc + dt < n:
                data[loc + dt, affected] += np.random.randn(n_affected) * 2.0
                labels[loc + dt] = True

    return data, labels, "Coherence Loss"


def scenario_drift_spatial(n=1200):
    """
    Spatial distribution drifts: the wavefront pattern slowly changes
    direction and frequency. Anomalies are spikes ON TOP of the drift.
    Living weights should track the drift; frozen detectors can't.
    """
    fg = FieldGenerator(4, 0.03, 0.06)
    t = np.arange(n, dtype=float)
    data = np.zeros((n, fg.n_channels))
    labels = np.zeros(n, dtype=bool)

    # Drifting wavefront
    for i in range(n):
        drift_angle = i / n * np.pi  # Direction rotates 180° over the signal
        drift_freq = 0.03 + 0.04 * (i / n)  # Frequency increases
        for ch, (px, py) in enumerate(fg.positions):
            phase = px * np.cos(drift_angle) + py * np.sin(drift_angle)
            data[i, ch] = np.sin(2*np.pi*drift_freq*i + phase * 2)
        data[i] += fg.noise * np.random.randn(fg.n_channels)

    # Inject anomalies
    for _ in range(15):
        loc = np.random.randint(150, n - 5)
        magnitude = np.random.uniform(3, 5)
        center = fg.positions[np.random.randint(fg.n_channels)]
        dists = np.sqrt(np.sum((fg.positions - center)**2, axis=1))
        data[loc] += magnitude * np.exp(-dists / 0.3) * np.random.choice([-1, 1])
        labels[loc] = True
        if loc + 1 < n:
            labels[loc + 1] = True

    return data, labels, "Spatial Drift"


# ================================================================
# EVALUATION
# ================================================================

def evaluate_spatial(detector, data, labels):
    """Run detector on spatial field data, compute AUROC."""
    n = len(data)
    scores = np.zeros(n)

    t0 = time.time()
    for i in range(n):
        scores[i] = detector.process(data[i])
    elapsed = time.time() - t0

    # Skip warmup
    w = 60
    scores = scores[w:]
    true = labels[w:]

    np_ = np.sum(true)
    nn = np.sum(~true)
    if np_ == 0 or nn == 0:
        return 0.5, 0.0, elapsed

    # AUROC via rank
    idx = np.argsort(-scores)
    sorted_labels = true[idx]
    tp = 0
    auc = 0
    for i in range(len(sorted_labels)):
        if sorted_labels[i]:
            tp += 1
        else:
            auc += tp
    auroc = auc / (np_ * nn)

    # Best F1
    best_f1 = 0
    for th in np.percentile(scores, np.linspace(50, 99, 40)):
        pred = scores > th
        tp_ = np.sum(pred & true)
        fp_ = np.sum(pred & ~true)
        fn_ = np.sum(~pred & true)
        pr = tp_ / (tp_ + fp_) if tp_ + fp_ > 0 else 0
        rc = tp_ / (tp_ + fn_) if tp_ + fn_ > 0 else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0
        best_f1 = max(best_f1, f1)

    return auroc, best_f1, elapsed


# ================================================================
# MAIN
# ================================================================

def main():
    gs = 4
    scenarios = [scenario_hotspot, scenario_wavefront_break,
                 scenario_coherence_loss, scenario_drift_spatial]

    detectors = [
        ("DeerskinField",    lambda: DeerskinFieldEngine(gs, 16, 0.008)),
        ("Channel Z-Score",  lambda: ChannelwiseZScore(gs, 50)),
        ("PCA",              lambda: PCADetector(gs, 100, 4)),
        ("Coherence",        lambda: SpatialCoherenceDetector(gs, 30)),
        ("MLP-AE (frozen)",  lambda: FrozenMLP(gs, 12)),
        ("MLP-AE (online)",  lambda: FlattenedMLP(gs, 12, 0.01)),
    ]

    n_trials = 8

    print("=" * 78)
    print("  DEERSKIN FIELD ENGINE: Spatiotemporal Anomaly Detection")
    print("  Where the input IS a 2D field — no artificial folding")
    print("=" * 78)

    print(f"\n  Grid: {gs}×{gs} = {gs*gs} channels")
    print(f"  Trials per scenario: {n_trials}")
    print(f"\n  PARAMETERS:")
    for name, build_fn in detectors:
        d = build_fn()
        print(f"    {name:<22} {d.n_params:>5} params")

    all_results = {}

    for scenario_fn in scenarios:
        _, _, sname = scenario_fn()
        print(f"\n  ── {sname} {'─' * (58 - len(sname))}")

        scenario_results = {}
        for det_name, build_fn in detectors:
            aurocs, f1s, times = [], [], []
            for trial in range(n_trials):
                np.random.seed(trial * 137 + hash(sname) % 10000)
                data, labels, _ = scenario_fn()
                det = build_fn()
                a, f, t = evaluate_spatial(det, data, labels)
                aurocs.append(a)
                f1s.append(f)
                times.append(t)

            scenario_results[det_name] = {
                'auroc': np.mean(aurocs),
                'std': np.std(aurocs),
                'f1': np.mean(f1s),
                'time': np.mean(times),
            }

        all_results[sname] = scenario_results

        # Print scenario results
        best_auroc = max(r['auroc'] for r in scenario_results.values())
        print(f"    {'Detector':<22} {'AUROC':>8} {'±':>6} {'F1':>7} {'Time':>8}")
        print(f"    {'─'*22} {'─'*8} {'─'*6} {'─'*7} {'─'*8}")
        for det_name, r in scenario_results.items():
            mark = " ◀" if r['auroc'] >= best_auroc - 0.01 else ""
            print(f"    {det_name:<22} {r['auroc']:>7.3f} {r['std']:>5.3f}"
                  f" {r['f1']:>6.3f} {r['time']:>6.2f}s{mark}")

    # Aggregate
    det_names = [n for n, _ in detectors]
    scenario_names = list(all_results.keys())

    print(f"\n{'=' * 78}")
    print(f"  AGGREGATE RESULTS")
    print(f"{'=' * 78}")

    hdr = f"    {'Detector':<22}"
    for sn in scenario_names:
        hdr += f" {sn[:11]:>11}"
    hdr += f" {'MEAN':>8}"
    print(hdr)
    print(f"    {'─'*22}" + f" {'─'*11}" * len(scenario_names) + f" {'─'*8}")

    means = {}
    for dn in det_names:
        row = f"    {dn:<22}"
        vals = []
        for sn in scenario_names:
            a = all_results[sn][dn]['auroc']
            vals.append(a)
            row += f" {a:>11.3f}"
        m = np.mean(vals)
        means[dn] = m
        row += f" {m:>8.3f}"
        print(row)

    # Wins per scenario
    wins = {dn: 0 for dn in det_names}
    for sn in scenario_names:
        best = max(all_results[sn][dn]['auroc'] for dn in det_names)
        for dn in det_names:
            if all_results[sn][dn]['auroc'] >= best - 0.015:
                wins[dn] += 1

    best_det = max(means, key=means.get)
    print(f"\n  WINS:")
    for dn in det_names:
        mk = " ◀ best overall" if dn == best_det else ""
        print(f"    {dn:<22} {wins[dn]}/{len(scenario_names)}{mk}")

    print(f"\n  BEST: {best_det} (mean AUROC: {means[best_det]:.3f})")

    # Analysis
    de = means.get('DeerskinField', 0)
    print(f"""
{'=' * 78}
  ANALYSIS
{'=' * 78}

  DeerskinField mean AUROC: {de:.3f}
  Best baseline mean AUROC: {means[best_det]:.3f} ({best_det})
  DeerskinField params:     {DeerskinFieldEngine(gs).n_params}
  MLP-AE params:            {FlattenedMLP(gs).n_params}

  This is the domain the deerskin was built for: the input IS a 2D
  spatial field. Each sensor is a position. Each timestep is a spatial
  snapshot. The moiré between the field and the membrane geometry
  computes spatial features that per-channel detectors cannot see.

  Key scenarios:
  - Spatial Hotspot: localized anomalies with spatial extent
  - Wavefront Break: disruptions to propagation patterns
  - Coherence Loss: spatially correlated channels becoming independent
  - Spatial Drift: slowly changing spatial distribution + anomalies

  The coherence loss scenario is the critical test — per-channel
  detectors (Z-Score) process each channel independently and can't
  see that spatial relationships have broken. The DeerskinField
  engine processes the spatial pattern as a whole.
""")


if __name__ == "__main__":
    main()
