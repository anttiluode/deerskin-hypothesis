# DeerskinEngine: Where Geometric Computation Meets Practical Application

## What This Is

An honest attempt to build a practical anomaly detector from the deerskin hypothesis primitives — living weights, mixed membrane geometries, frequency multiplexing, and carrier wave encoding — and benchmark it against standard approaches.

**The headline result: it doesn't win on 1D time series.** And that's the most important finding.

## The Benchmark

Six detectors, four scenarios, eight trials each. All numpy-only.

| Detector | Spikes | Drift | Phase Shift | Multi-Scale | **Mean AUROC** | Params |
|----------|--------|-------|-------------|-------------|----------------|--------|
| **DeerskinEngine** | 0.767 | **0.829** | 0.469 | 0.531 | 0.649 | **24** |
| Z-Score | 0.829 | 0.722 | 0.473 | 0.523 | 0.637 | 1 |
| EWMA | 0.684 | 0.665 | 0.448 | 0.517 | 0.579 | 2 |
| IsolationForest | 0.879 | 0.749 | 0.511 | 0.563 | 0.675 | 3 |
| MLP-AE (frozen) | **0.962** | 0.909 | 0.713 | **0.891** | 0.869 | 144 |
| MLP-AE (online) | 0.959 | **0.964** | **0.740** | 0.816 | **0.870** | 144 |

## Where It Wins (and Doesn't)

**DeerskinEngine wins on Concept Drift vs all non-MLP baselines:**
- DeerskinEngine 0.829 vs Z-Score 0.722 (+15%)
- DeerskinEngine 0.829 vs IsolationForest 0.749 (+11%)
- DeerskinEngine 0.829 vs EWMA 0.665 (+25%)

The living weights genuinely track distributional change better than statistical baselines. 24 geometric parameters adapting during inference outperform 3 statistical parameters that can't adapt.

**But the MLP autoencoders (6× more parameters) still dominate.** On 1D time series, there isn't enough spatial structure for the geometric substrate to exploit. A signal folded into a 16×16 grid is artificial — the membrane can't see meaningful 2D patterns in 1D data.

## Why This Matters

The deerskin primitives were designed for spatiotemporal computation:
- The MoiréFormer beat Transformers 3-1 on **sequence tasks** (inherently structured)
- MoiréNet solved XOR at 100% vs MLP's 30% on **logic tasks** (inherently geometric)
- The geometry zoo showed different cell types specialize for **different tasks**

These wins happen when the input is **inherently multi-dimensional and structured**. 1D scalars don't have enough structure to reward geometric encoding.

## The Right Application

The DeerskinEngine architecture is designed for:

- **Multi-channel EEG**: Each electrode IS a spatial position. The signal IS already a 2D field.
- **Multi-sensor industrial monitoring**: Sensor array on a surface — natural 2D structure.
- **High-frequency trading**: Order book is a 2D field (price × time) with natural geometry.
- **Video anomaly detection**: Frames are 2D. Temporal changes create 3D structure.

In all these cases:
1. The geometric substrate can exploit real spatial patterns (no artificial folding)
2. Different membrane geometries match different signal types naturally
3. Living weights track changing system states
4. Frequency multiplexing captures phenomena at multiple scales simultaneously

## Files

- `deerskin_engine.py` — Complete benchmark: engine + 5 baselines + 4 scenarios + analysis
- `deerskin_engine_demo.html` — Interactive visualization of the engine processing signals

## Running

```bash
python deerskin_engine.py
```

No dependencies beyond numpy. Runs in ~30 seconds.

## Connection to the Deerskin Hypothesis

This is the "application test" for the theoretical framework. The result:

**The computational primitive is richer (proven by MoiréFormer, carrier wave encoding, frequency multiplexing experiments). But primitive richness only matters when the input structure can exploit it.**

Applying geometric computation to 1D data is like using a microscope to read a billboard — the tool is powerful but the problem doesn't need it. The deerskin primitives want spatiotemporal fields, not scalars. That's where the 12 channels per connection, the living weight adaptation, and the multi-geometry ensemble can actually compute something a scalar weight can't.
