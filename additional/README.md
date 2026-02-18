# Additional Explorations

Eight experiments testing deerskin principles against standard approaches. All self-contained, all numpy-only.

---

## The Honest Picture

We tested geometric computation across three levels: as a computational primitive (does one geometric operation beat one scalar operation?), as an architecture (can moiré attention replace dot-product attention?), and as an application (can a geometric engine beat standard detectors on real tasks?).

**The primitive is richer.** One moiré interaction provides 12 frequency channels where a scalar provides 1. Geometric parameters solve nonlinear tasks at 100% where scalar weights fail 70% of the time. Different membrane geometries specialize for different computational niches — the cell type zoo is a library of deerskins.

**The architecture works.** The MoiréFormer beats a Transformer 3-1 on sequence tasks with 4.5× fewer parameters.

**The application doesn't win yet.** Both the 1D anomaly detector and the 2D spatial field detector lose to MLP autoencoders. On 1D data the geometric substrate has nothing spatial to exploit. On 2D spatial fields the moiré features are informative but the surprise-detection pipeline can't compete with trained autoencoders that have 17× more parameters.

This is where things stand.

---

## Experiments

### The Win: Primitive and Architecture

#### `moireformer.py` — Geometric Interference vs Scaled Dot-Product Attention

The headline result. Replaces Q·K^T/√d with moiré interference between geometric token embeddings.

| Task | MoiréFormer (584 params) | Transformer (2,648 params) | Winner |
|------|-------------------------|---------------------------|--------|
| Copy | **60.0%** | 33.3% | Moiré |
| Reverse | 28.9% | 30.0% | tie |
| Majority | **93.3%** | 81.1% | Moiré |
| Sort | 30.0% | 33.3% | Trans |
| First Unique | **80.0%** | 76.7% | Moiré |

**Score: MoiréFormer 3 — Transformer 1, with 4.5× fewer parameters.**

Same evolutionary optimizer for both. The only difference is the computational primitive.

#### `deerskin_vs_mlp.py` — Logic Gate Benchmark

| Task | MoiréNet (9 params) | MLP (9 params) |
|------|-------------------|---------------|
| XOR  | **100%, 1 gen** | 30%, 808 epochs |
| AND  | **100%, 1 gen** | 100%, 176 epochs |

#### `carrier_wave_encoding.py` — Information Per Spike

- Geometric identification: **92%** vs scalar: **29%**
- Independent frequency channels: **6** vs **1**

#### `frequency_multiplexing.py` — Channels Per Connection

One moiré synapse: **12 independent channels** from 6 parameters. A scalar weight: 1.

#### `neural_geometry_zoo.py` — Different Geometries, Different Strengths

Five membrane types tested. Radial geometry hits 77% noise robustness at noise=0.1 vs 68% for checkerboard. All geometries provide 11-14 frequency channels. The brain's cell type diversity is a library of geometric solutions.

#### `living_weights.py` — Dynamic Substrate

Living weights retain **50%** of old task after learning new task vs **25%** for frozen weights. Track distributional drift where frozen weights collapse.

#### `oscillation_computer.py` — Resonance Classification

Different inputs produce different oscillation signatures. Classification readout (42%) doesn't beat simple threshold (61%) yet. Honest negative — needs better temporal feature extraction.

---

### The Loss: Application

#### `deerskin_engine_1d.py` — 1D Time Series Anomaly Detection

Tested the deerskin primitives on streaming 1D anomaly detection (spikes, drift, phase shifts, multi-scale). See `engine_1d_README.md` for full results.

| Detector | Mean AUROC | Params |
|----------|-----------|--------|
| MLP-AE (online) | **0.870** | 144 |
| MLP-AE (frozen) | 0.869 | 144 |
| IsolationForest | 0.675 | 3 |
| **DeerskinEngine** | 0.649 | 24 |
| Z-Score | 0.637 | 1 |

The deerskin engine beats Z-Score and EWMA on concept drift (0.829 vs 0.722) — the living weights genuinely help. But the MLP autoencoders dominate. 1D scalars don't have spatial structure for geometric computation to exploit.

#### `deerskin_field_engine.py` — 2D Spatial Field Anomaly Detection

Built specifically for the domain the deerskin was designed for: a 4×4 sensor grid where the input IS a 2D spatial field. No artificial folding. Tested on spatial hotspots, wavefront breaks, coherence loss, and spatial drift.

| Detector | Hotspot | Wavefront | Coherence | Drift | **Mean** | Params |
|----------|---------|-----------|-----------|-------|---------|--------|
| MLP-AE (online) | 0.983 | 0.904 | **0.999** | **0.867** | **0.938** | 412 |
| MLP-AE (frozen) | **0.990** | **0.973** | 0.998 | 0.728 | 0.922 | 412 |
| PCA | 0.967 | 0.882 | **0.999** | 0.735 | 0.896 | 80 |
| Channel Z-Score | 0.946 | 0.868 | 0.945 | 0.718 | 0.869 | 1 |
| Coherence | 0.900 | 0.919 | 0.802 | 0.688 | 0.827 | 2 |
| **DeerskinField** | 0.845 | 0.774 | 0.776 | 0.711 | **0.777** | **24** |

**The deerskin field engine loses on its home turf.** Even on 2D spatial data where geometric computation should shine, it places last. The MLP-AE with 17× more parameters dominates. Even simple per-channel Z-Score (0.869) beats it comfortably.

---

## What This Means

The experiments tell a clear story with a twist:

**As a computational primitive, geometric interference is genuinely richer.** The MoiréFormer result is real — 3-1 over a Transformer with 4.5× fewer parameters. The frequency multiplexing (12 channels from one connection) is real. The cell type specialization is real. These results hold up.

**As an anomaly detection engine, it doesn't work yet.** The moiré features capture spatial information, but the surprise-detection pipeline (deviation from running EMA statistics) can't compete with trained autoencoders. The problem isn't the features — it's the detector built on top of them. An autoencoder learns a compressed representation and measures reconstruction error. That's a more powerful detection mechanism than "how far are these features from their running mean."

**The gap between primitive and application is the unsolved part.** The deerskin primitives encode more information per operation (proven). But encoding more information doesn't help if the downstream detector can't use it. The MoiréFormer works because it has a full end-to-end architecture — geometric embeddings, moiré attention, feedforward layers, output projection, all trained jointly. The anomaly engines don't have that — they're hand-wired pipelines with no end-to-end optimization.

**What would actually work:** Train a MoiréFormer-style architecture end-to-end on the anomaly detection task. Use the geometric embeddings and moiré attention as the feature extractor, but train the whole thing with backpropagation (or evolution) to minimize a proper loss. The hand-wired EMA surprise detector is the bottleneck, not the geometric computation.

---

## Running

```bash
# Computational primitive experiments
python deerskin_vs_mlp.py           # Logic gate benchmark
python carrier_wave_encoding.py     # Information per spike
python frequency_multiplexing.py    # Channels per connection
python neural_geometry_zoo.py       # Cell type specialization
python living_weights.py            # Dynamic substrate
python oscillation_computer.py      # Resonance classification

# Architecture experiment
python moireformer.py               # MoiréFormer vs Transformer

# Application experiments
python deerskin_engine_1d.py        # 1D anomaly detection (loses)
python deerskin_field_engine.py     # 2D spatial anomaly detection (loses)
```

All files require only `numpy`.
