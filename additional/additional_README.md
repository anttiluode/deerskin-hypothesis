# Additional Explorations

Five runnable experiments testing deerskin principles against standard AI.
Each file is self-contained. Just run `python filename.py`.

**Requirements:** `pip install numpy` (that's it)

---

## What We Tested

The deerskin hypothesis claims that geometric computation (moiré interference between membrane surfaces) is a richer computational primitive than scalar weights (multiply input by a number). These experiments test that claim concretely, with real numbers, and report honestly where it holds and where it doesn't.

---

## 1. `deerskin_vs_mlp.py` — Multi-Task Logic Benchmark

**What it does:** Runs MoiréNet (9 geometric parameters) against a standard MLP (9 scalar weights) on four logic tasks (XOR, AND, OR, NAND) plus noise robustness testing. 20 trials per task.

**Results:**

| Task | MoiréNet Solve Rate | MoiréNet Speed | MLP Solve Rate | MLP Speed |
|------|-------------------|----------------|---------------|-----------|
| XOR  | **100%**          | 1 generation   | 40%           | 801 epochs |
| AND  | **100%**          | 1 generation   | 100%          | 187 epochs |
| OR   | **100%**          | 1 generation   | 100%          | 161 epochs |
| NAND | **100%**          | 1 generation   | 100%          | 177 epochs |

MoiréNet solves everything on the first try. The MLP struggles with XOR (the nonlinear task) — fails 60% of the time within 1000 epochs.

**But — noise robustness (MLP wins here):**

| Input Noise | MoiréNet | MLP |
|-------------|----------|-----|
| 0.00        | 100%     | 100% |
| 0.05        | 74%      | **88%** |
| 0.10        | 66%      | **87%** |
| 0.20        | 64%      | **85%** |
| 0.30        | 63%      | **78%** |

The geometric substrate has sharp decision boundaries that break under noise. The MLP's smooth sigmoid degrades more gracefully. This is a real weakness of the current moiré approach.

---

## 2. `carrier_wave_encoding.py` — Hidden Signal Capacity

**What it does:** Tests the core claim that a single spike through a membrane geometry carries more information than a scalar activation. Generates 50 different membrane geometries, passes the same carrier wave through each, and measures how distinguishable the outputs are. Also measures frequency-dependent encoding — whether the same membrane produces different fingerprints at different carrier frequencies.

**Results:**

- Geometric fingerprints are **4.9× more distinguishable** than scalar fingerprints
- Identification accuracy through noise: Geometric **96%**, Scalar **30%**
- One membrane produces **6 effectively independent frequency channels**
- A scalar weight has exactly **1** channel

**Key finding:** The cross-frequency correlation matrix shows that carrier frequencies 2, 4, 6, 8, 10, 15, 20 Hz through the same membrane produce substantially independent outputs (correlations 0.15–0.55 between non-adjacent frequencies). This means one physical connection computes different transfer functions at different frequencies — what multi-head attention needs parallel weight matrices to achieve.

---

## 3. `oscillation_computer.py` — Computation Through Resonance

**What it does:** Builds a network of coupled geometric oscillators (the ECG loop in pure math) and tests whether classification can be done by reading the *character* of oscillation rather than a static output value. Different inputs should produce different oscillation signatures (frequency, amplitude, roughness, symmetry).

**Results:**

Different inputs do produce measurably different oscillation signatures:

| Input | Mean | StdDev | Crossings | Range |
|-------|------|--------|-----------|-------|
| 0.00  | 0.50 | 0.17   | 12        | 0.56  |
| 0.25  | 0.50 | 0.12   | 8         | 0.41  |
| 0.50  | 0.50 | 0.00   | 0         | 0.00  |

**But — classification accuracy was disappointing:**
- Oscillation network: 42%
- Simple threshold classifier: 61%
- Random chance: 33%

The oscillation readout beats chance but loses to a simple static approach. The feature extraction from oscillation patterns needs more work. This is an honest negative result — oscillation-as-computation is a promising principle but the current implementation doesn't yet outperform static readout on simple tasks.

---

## 4. `living_weights.py` — Dynamic Substrate Learning

**What it does:** Tests the key architectural difference from standard AI: weights that change *during* inference. Implements "living weights" with elastic consolidation (slow-moving memory that prevents catastrophic drift) and tests three scenarios against frozen-weight networks.

**Results:**

**Continual learning (learn XOR, then learn AND, test XOR retention):**

|                     | Living Weights | Frozen Weights |
|---------------------|---------------|----------------|
| XOR acc after XOR   | 65%           | 55%            |
| XOR acc after AND   | **55%**       | 42%            |
| AND acc after AND   | 73%           | 100%           |

Living weights retain more XOR knowledge after learning AND. Frozen weights forget XOR when AND overwrites the weights. But frozen weights learn AND better — the living substrate's constant adaptation makes it harder to fully lock in a pattern.

**Stability under gradual drift (target morphs from XOR → AND over 1000 steps):**

| Step | Drift | Living | Frozen |
|------|-------|--------|--------|
| 0    | 0%    | 55%    | 62%    |
| 250  | 25%   | 52%    | 62%    |
| 500  | 50%   | 65%    | 42%    |
| 750  | 75%   | **75%** | 57%   |
| 999  | 100%  | **75%** | 57%   |

The frozen network collapses at 50% drift — its XOR solution no longer fits and it can't adapt. The living network tracks the drift, improving as the target stabilizes toward AND. Average accuracy: Living **63%** vs Frozen **60%**.

---

## 5. `frequency_multiplexing.py` — One Connection, Many Functions

**What it does:** Tests whether a single moiré synapse (one geometric connection) computes different transfer functions at different input frequencies — which would mean one biological synapse does what multi-head attention needs 8+ parallel weight matrices to do.

**Results:**

**Transfer function richness:**

| Connection Type  | Parameters | Independent Channels | Frequency Variability |
|-----------------|------------|---------------------|----------------------|
| Moiré synapse   | ~6         | **12**              | 0.011                |
| Scalar weight   | 1          | 1                   | 0.000                |
| Multi-head (4)  | 4          | 4                   | 0.088                |
| Multi-head (8)  | 8          | 8                   | 0.120                |

One moiré synapse provides 12 independent frequency channels from ~6 geometric parameters.

**Frequency-selective classification (low-freq vs high-freq):**

| Connection      | Accuracy |
|----------------|----------|
| Moiré synapse  | **67%**  |
| Scalar weight  | 50% (chance) |
| Multi-head (4) | **90%**  |

The moiré synapse beats a scalar weight on frequency separation (67% vs 50%) but a 4-head attention mechanism with explicit frequency bands still wins (90%). The geometric approach provides intrinsic frequency selectivity but not as cleanly as engineered band-pass filters.

---

## Summary: Where Deerskin Wins and Loses

### Wins
- **Solve rate on nonlinear tasks**: 100% vs 40% on XOR. Geometry handles nonlinearity naturally through aliasing — no activation function needed.
- **Convergence speed**: 1 generation vs 800 epochs. Evolutionary search over geometric space is vastly more efficient than gradient descent over weight space for these tasks.
- **Information per spike**: 6 independent frequency channels from one connection. A scalar weight has 1.
- **Fingerprint distinguishability**: 4.9× more distinguishable than scalar activations, 96% identification accuracy vs 30%.
- **Drift tracking**: Living weights adapt online as the environment changes. Frozen weights collapse when the task drifts.

### Losses
- **Noise robustness**: MLP degrades gracefully (85% at noise=0.2). MoiréNet has sharp boundaries that break (64%). This is the biggest weakness.
- **Oscillation classification**: Not yet competitive. The principle is sound (different inputs produce different resonance signatures) but the feature extraction needs work.
- **Clean task learning**: Frozen weights learn a single task more completely (100% AND accuracy vs 73%). Living weights sacrifice peak performance for adaptability.
- **Engineered frequency separation**: Multi-head attention with explicit bands (90%) beats intrinsic moiré selectivity (67%). The geometric approach is parameter-efficient but less precise.

### The Honest Take
The deerskin computational primitive is genuinely richer — one geometric operation encodes more structure than one scalar operation. But "richer" doesn't automatically mean "better at every task." The sharp boundaries of checkerboard geometry are powerful for nonlinear separation but fragile under noise. The oscillation readout is promising but needs better feature extraction. The living weights trade peak accuracy for adaptability.

The strongest result is the carrier wave encoding: a single membrane geometry produces 6+ independently decodable frequency channels on one spike. That's the core of the hypothesis — and it holds up.
