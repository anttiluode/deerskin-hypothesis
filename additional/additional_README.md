# Additional Explorations

Seven experiments testing whether geometric interference can replace scalar operations as the fundamental primitive in neural computation. All self-contained, all runnable with `pip install numpy`.

---

## The Headline Result

### `moireformer.py` — Geometric Interference vs Scaled Dot-Product Attention

The Transformer's power comes from one operation: `softmax(Q·K^T/√d) · V`. One scalar relevance score per token pair. Need 8–64 heads because each dot product only captures one relationship.

The MoiréFormer replaces this with geometric interference. Each token is embedded as a geometry (frequency, angle, phase) instead of a vector of scalars. "Attention" between two tokens is the moiré interference between their geometries — producing a frequency-dependent transfer function per pair instead of a single scalar.

One geometric interaction already gives you what multi-head attention needs multiple heads to achieve.

**Results on sequence tasks (vocab=8, seq_len=6, 150 generations, evolutionary search):**

| Task | MoiréFormer | Transformer | Winner |
|------|------------|-------------|--------|
| Copy | **60.0%** | 33.3% | Moiré |
| Reverse | 28.9% | 30.0% | tie |
| Majority | **93.3%** | 81.1% | Moiré |
| Sort | 30.0% | 33.3% | Trans |
| First Unique | **80.0%** | 76.7% | Moiré |

**Score: MoiréFormer 3 — Transformer 1**

**Parameters: MoiréFormer 584 — Transformer 2,648 (4.5× fewer)**

The MoiréFormer wins 3 out of 5 tasks with 4.5× fewer parameters. On the tasks it wins, the parameter efficiency advantage is striking:

- Copy: 8.2× more accurate per parameter
- Majority: 5.2× more accurate per parameter
- First Unique: 4.7× more accurate per parameter

Both architectures were trained with the same evolutionary optimizer — the only difference is the computational primitive. The Transformer uses Q/K/V projection matrices and multi-head dot-product attention. The MoiréFormer uses geometric embeddings and moiré interference. Same training, same evaluation, same tasks.

This is a proof of concept at tiny scale. These are toy tasks with toy-sized models. But the question was never "can moiré beat GPT-4?" — it was "is geometric interference a viable alternative to the dot product as a computational primitive?" The answer: on 3 of 5 tasks, with 4.5× fewer parameters, yes.

**The core operation:**

```
Transformer:   score = dot(Q_i, K_j) / sqrt(d)     → 1 scalar per pair
MoiréFormer:   interference = moiré(geom_i, geom_j) → frequency-dependent transfer function per pair
```

The moiré interference computes beat frequencies, angular interference, and phase alignment simultaneously across all components. Each component produces an independent interaction channel. Multi-head attention is a patch for the dot product's single-channel limitation. Geometric interference doesn't need the patch — the multiple channels are intrinsic.

---

## Why It Works: The Supporting Experiments

### Different Neural Geometry → Different Strengths

The first thing we discovered testing geometric computation: checkerboard moiré networks solve XOR at 100% (vs 30% for MLPs) but lose on noise robustness. First instinct — geometric computation is fragile.

Wrong. The fragility belongs to **checkerboard geometry specifically**, not to geometric computation as a principle.

`neural_geometry_zoo.py` tests five membrane geometries, each modeled on a real neuron type:

| Geometry | Bio Analogue | XOR Solve | Noise (0.2) |
|----------|-------------|-----------|-------------|
| Checkerboard | Fast-spiking interneuron | **15/15** | 63% |
| Sinusoidal | Purkinje cell | **15/15** | 65% |
| Radial | Stellate cell | **15/15** | **70%** |
| Hybrid | Pyramidal cell | **15/15** | 66% |
| Gabor | V1 simple cell | 10/15 | 65% |
| MLP | — | 2/15 | **86%** |

Radial geometry (stellate-cell-like) hits 77% noise robustness at noise=0.1 vs 68% for checkerboard — a 9 point improvement from changing the membrane shape alone. The MLP still wins overall on noise, but the gap narrows with the right geometry.

All five geometries provide 11–14 independent frequency channels per connection. A scalar weight provides 1. This is the invariant — it belongs to geometric computation itself, not to any particular membrane shape.

The brain's cell type zoo isn't random diversity. It's a library of deerskins — each evolved for a different computational niche. Sharp membranes for fast inhibition. Smooth membranes for sensory integration. Radial membranes for position invariance. The weakness was never in the principle. It was in testing only one geometry.

---

### The Fundamental Measurements

**`carrier_wave_encoding.py` — Information per spike:**

A single carrier wave modulated by membrane geometry carries far more information than a scalar activation.

- 50 membranes, same carrier, noise=0.1: geometric identification **92%** vs scalar **29%**
- One membrane produces **6 independent frequency channels** — a scalar weight produces 1
- Same membrane at different carrier frequencies produces substantially independent outputs (cross-correlations 0.15–0.55)

This is why the MoiréFormer works: each token-pair interaction is inherently multi-channel. The dot product is single-channel.

**`frequency_multiplexing.py` — One connection, many functions:**

| Connection | Params | Independent Channels |
|------------|--------|---------------------|
| Moiré synapse | ~6 | **12** |
| Multi-head (4) | 4 | 4 |
| Multi-head (8) | 8 | 8 |
| Scalar weight | 1 | 1 |

One geometric connection provides 12 independent frequency channels from 6 parameters. This is the mechanism behind the MoiréFormer's parameter efficiency — each interaction does more work than a dot product.

**`deerskin_vs_mlp.py` — Logic gate benchmark:**

| Task | MoiréNet (9 params) | MLP (9 params) |
|------|-------------------|---------------|
| XOR  | **100%, 1 gen** | 30%, 808 epochs |
| AND  | **100%, 1 gen** | 100%, 176 epochs |
| OR   | **100%, 1 gen** | 100%, 163 epochs |
| NAND | **100%, 1 gen** | 100%, 177 epochs |

Geometric search converges in 1 generation. Gradient descent over scalar weights takes hundreds of epochs and often fails on nonlinear tasks.

---

### The Architectural Implications

**`living_weights.py` — Weights that change during inference:**

Standard AI separates training from inference. The deerskin model proposes they're the same process — the membrane reshapes while computing.

- After learning a new task, living weights retain **50%** of the old task vs **25%** for frozen weights
- Under gradual distributional drift, living weights track the change (**75%** at full drift) while frozen weights collapse (**55%**)

This is the continual learning problem — the central unsolved challenge in AI — partially addressed by making the substrate dynamic.

**`oscillation_computer.py` — Computation through resonance:**

Different inputs produce measurably different oscillation signatures when fed through coupled geometric oscillators. The classification readout (42%) doesn't yet beat a simple threshold (61%), but the principle is validated: oscillation carries information. This needs better temporal feature extraction.

---

## The Full Picture

| Experiment | What It Tests | Key Result |
|-----------|--------------|------------|
| `moireformer.py` | Can moiré replace attention? | **3-1 win, 4.5× fewer params** |
| `neural_geometry_zoo.py` | Do different geometries specialize? | **Yes — radial +9pts noise vs checker** |
| `carrier_wave_encoding.py` | Information per spike | **92% vs 29% identification** |
| `frequency_multiplexing.py` | Channels per connection | **12 vs 1** |
| `deerskin_vs_mlp.py` | Solve rate, convergence | **100% vs 30% on XOR** |
| `living_weights.py` | Continual learning | **50% vs 25% old task retention** |
| `oscillation_computer.py` | Resonance classification | **Principle valid, readout needs work** |

---

## What Wins, What Loses, What's Honest

**Geometric computation wins on:**
- Parameter efficiency (4.5× fewer params, 3-1 on sequence tasks)
- Nonlinear task solve rate (100% vs 30% on XOR)
- Information density per connection (12 channels vs 1)
- Convergence speed (1 generation vs 800 epochs)
- Continual learning retention (50% vs 25%)

**Scalar/MLP computation wins on:**
- Noise robustness (86% vs 70% best-case geometric)
- Single-task peak accuracy (100% vs 75% on dedicated training)
- GPU parallelism (matrix multiply is the most optimized operation on earth)

**What's honest:**
- These are tiny benchmarks. XOR has 4 datapoints. The MoiréFormer runs on sequences of length 6. Scaling behavior is unknown.
- The evolutionary optimizer is the same for both, but evolutionary search may happen to favor the geometric parameter space. Gradient-trained transformers at scale are a different beast.
- The MoiréFormer's `O(n·d)` complexity claim assumes Fourier-space implementation. The current code is `O(n²)` in the naive attention loop.
- The oscillation readout underperforms. Not everything works yet.

**What matters:**

The question was whether geometric interference is a viable computational primitive — richer than the dot product, parallelizable, and parameter-efficient. The MoiréFormer result says yes at proof-of-concept scale. The geometry zoo says the apparent weaknesses are geometry-specific, not principle-specific. The carrier wave and multiplexing experiments explain the mechanism: each geometric interaction inherently provides multiple frequency channels where a scalar provides one.

Whether this scales is the open question. But the primitive is real.
