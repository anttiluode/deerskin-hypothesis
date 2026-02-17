"""
MoiréNet: Geometry-Based Neural Computing
==========================================
Computation via geometric interference between discrete grids.
No stored weight matrix. Each "neuron" is defined by 3 parameters:
(frequency, angle, phase) — its membrane geometry.

The "weight" between two neurons is the moiré interference pattern
between their grids, computed on-the-fly. This is O(N) storage
instead of O(N²).

Part of the Deerskin Hypothesis project.
"""

import numpy as np


class MoireNeuron:
    """A single geometric neuron defined by its grid parameters."""

    def __init__(self, freq=10.0, angle=0.0, phase=0.0, grid_size=32):
        self.freq = freq
        self.angle = angle
        self.phase = phase
        self.grid_size = grid_size
        self._grid = None
        self._update_grid()

    def _update_grid(self):
        """Generate the 2D checkerboard pattern for this neuron."""
        s = self.grid_size
        y, x = np.mgrid[0:s, 0:s]
        cx, cy = s / 2, s / 2

        # Rotate coordinates
        c, sn = np.cos(self.angle), np.sin(self.angle)
        rx = (x - cx) * c - (y - cy) * sn + self.phase
        ry = (x - cx) * sn + (y - cy) * c

        # Checkerboard (XOR of floor divisions)
        cell = max(1, s / (self.freq + 1e-6))
        gx = np.floor(rx / cell).astype(int) % 2
        gy = np.floor(ry / cell).astype(int) % 2
        self._grid = (gx ^ gy).astype(np.float32)

    @property
    def grid(self):
        return self._grid

    def set_params(self, freq, angle, phase):
        self.freq = freq
        self.angle = angle
        self.phase = phase
        self._update_grid()


class MoireNet:
    """
    A network of MoiréNeurons.

    Architecture: 3 hidden neurons + 1 output combination.
    Each hidden neuron generates a checkerboard pattern.
    Input is encoded as a position on the pattern.
    Output is the moiré interference at that position.

    Total parameters: 9 (3 per hidden neuron × 3 neurons)
    """

    def __init__(self, n_hidden=3, grid_size=32):
        self.n_hidden = n_hidden
        self.grid_size = grid_size
        self.neurons = [
            MoireNeuron(
                freq=np.random.uniform(5, 20),
                angle=np.random.uniform(0, np.pi),
                phase=np.random.uniform(0, 2 * np.pi),
                grid_size=grid_size
            )
            for _ in range(n_hidden)
        ]

    def forward(self, x):
        """
        Forward pass: evaluate input point(s) on the moiré field.

        Args:
            x: array of shape (N, 2) with values in [0, 1]

        Returns:
            array of shape (N,) with predictions in [0, 1]
        """
        x = np.atleast_2d(x)
        results = np.zeros(len(x))

        for xi in range(len(x)):
            px = int(x[xi, 0] * (self.grid_size - 1))
            py = int(x[xi, 1] * (self.grid_size - 1))
            px = np.clip(px, 0, self.grid_size - 1)
            py = np.clip(py, 0, self.grid_size - 1)

            # Sample each neuron's grid at the input position
            # and compute moiré (XOR-like combination)
            vals = [n.grid[py, px] for n in self.neurons]

            # Moiré combination: parity of activated grids
            result = sum(vals) % 2
            results[xi] = result

        return results

    def get_params(self):
        """Get all 9 parameters as a flat array."""
        params = []
        for n in self.neurons:
            params.extend([n.freq, n.angle, n.phase])
        return np.array(params)

    def set_params(self, params):
        """Set all 9 parameters from a flat array."""
        for i, n in enumerate(self.neurons):
            n.set_params(
                freq=params[i * 3],
                angle=params[i * 3 + 1],
                phase=params[i * 3 + 2]
            )

    def evolve_for_xor(self, generations=200, population=50):
        """
        Evolutionary optimization to solve XOR.

        Uses geometric parameter search (not gradient descent).
        This is analogous to membrane remodeling — adjusting channel
        distributions rather than synaptic weights.

        Returns:
            (accuracy, generations_needed)
        """
        # XOR dataset
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        Y = np.array([0, 1, 1, 0], dtype=float)

        best_params = self.get_params()
        best_acc = 0

        for gen in range(generations):
            # Generate population of parameter variations
            pop = []
            for _ in range(population):
                noise = np.random.randn(9) * 0.5
                candidate = best_params + noise
                # Clamp to valid ranges
                for i in range(3):
                    candidate[i * 3] = np.clip(candidate[i * 3], 1, 30)        # freq
                    candidate[i * 3 + 1] = candidate[i * 3 + 1] % np.pi        # angle
                    candidate[i * 3 + 2] = candidate[i * 3 + 2] % (2 * np.pi)  # phase
                pop.append(candidate)

            # Evaluate each candidate
            for params in pop:
                self.set_params(params)
                preds = self.forward(X)
                acc = np.mean((preds > 0.5) == Y)
                if acc > best_acc:
                    best_acc = acc
                    best_params = params.copy()

            if best_acc >= 1.0:
                self.set_params(best_params)
                return 1.0, gen + 1

        self.set_params(best_params)
        return best_acc, generations


if __name__ == "__main__":
    net = MoireNet()
    acc, gens = net.evolve_for_xor()
    print(f"MoiréNet XOR: accuracy={acc:.0%}, generations={gens}")
    print(f"Parameters: {net.get_params()}")
