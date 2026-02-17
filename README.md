# 🦌 The Deerskin Hypothesis

Demo: https://anttiluode.github.io/deerskin-hypothesis/deerskin_interactive.html 

### Neuronal Membranes as Holographic Computational Surfaces

> *"The brain is not a computer calculating numbers. It is a hall of deerskins — each neuron a two-dimensional computational surface whose ion channel mosaic encodes geometric information."*

---

## What Is This?

This repository contains a **speculative working paper** proposing that the neuronal membrane — not just the soma or synapses — functions as a two-dimensional computational surface. In this framework:

- The **soma pulse** is a carrier wave, not a binary message
- The **membrane's ion channel mosaic** is a geometric encoding medium (the "deerskin")
- **Synaptic communication** is moiré interference between two membrane geometries
- **Neural oscillations** are the inevitable cost of finite-resolution self-observation
- **Memory** is the geometry itself, written into channel distributions

The hypothesis originated from an experimental accident in [PerceptionLab](https://github.com/anttiluode/perceptionlab) where a checkerboard pattern in a feedback loop spontaneously produced ECG-like oscillations.

## ⚠️ Status: Speculative

This is **not peer-reviewed science**. It is a working speculation that connects:
- Published literature on dendritic computation (Mannion & Kenyon, 2024; London & Häusser, 2005)
- Experimental results from minimal feedback simulations
- Holographic memory principles (Gabor, 1948; Pribram, 1971)
- The author's personal experience with altered perception after brain surgery

The paper is honest about what is established, what is speculative, and what would need to happen to test these ideas. See Section 8 (Limitations) for a frank assessment.

---

## Repository Structure

```
deerskin-hypothesis/
├── paper/
│   └── deerskin_hypothesis.md      # The full speculative paper
├── demos/
│   └── deerskin_interactive.html   # Interactive visualization of the core concepts
├── benchmark/
│   ├── moire_net.py                # MoiréNet: geometry-based neural network
│   └── benchmark_xor.py            # XOR benchmark: MoiréNet vs MLP
├── perceptionlab/
│   ├── workflows/
│   │   ├── ecg.json                # Original ECG emergence workflow
│   │   └── spinningecg.json        # Spinning grid variant
│   └── nodes/
│       ├── spinningcheckerboardnode.py
│       ├── coupler.py              # Homeostatic Coupler (the regulator)
│       ├── imagetovectornode.py     # The sampling operator
│       ├── vectorsplitternode.py    # Vector decomposition
│       ├── constantsignalnode.py
│       ├── phasespacenode.py        # Takens embedding
│       ├── eegprocessor.py
│       ├── EDF_EEG_loader.py
│       ├── signalamplifier.py
│       ├── complexinference.py
│       └── DisplayNode.py
├── janus/
│   ├── janus_brainv2.py            # Holographic memory (Spin Cortex)
│   └── brainmagnifierglass.py      # Resonance scanner / tomography
└── README.md
```

---

## Key Results

### 1. ECG Emergence from Geometric Feedback
A checkerboard → sampling → regulation feedback loop spontaneously produces cardiac-like oscillations. No biological components. Pure geometry + homeostasis.

**Critical finding:** Oscillation character depends on sampling resolution:
| Resolution | Behavior |
|---|---|
| 64 | Slow, simple |
| 128 | Heartbeat emerges |
| **256** | **Rich, lifelike ECG** |
| 1024 | System locks up |
| 2048 | Rhythm returns differently |

### 2. MoiréNet vs MLP on XOR
| Metric | MoiréNet | MLP |
|---|---|---|
| Parameters | 9 (geometric) | 9 (scalar weights) |
| Success rate | **30/30 (100%)** | 22/30 (73%) |
| Convergence | **19 generations** | 541 epochs |
| Time | **0.002s** | 0.318s |

### 3. Holographic Memory Storage
The Janus Brain stores multiple images in a single complex field addressed by frequency/angle/spin. Scanning across addresses produces resonance pulses — recognition events.

---

## The Core Argument in 60 Seconds

1. **Ion channels are discrete sampling points** on a continuous electrochemical field
2. **Different channel types cluster** in spatially heterogeneous patterns — a biological "checkerboard"
3. When neuron A **observes** neuron B through a synapse, the presynaptic release pattern is **sampled** by the postsynaptic receptor grid
4. This sampling introduces **moiré interference** — the "computation" at the synapse
5. In a feedback loop, moiré mismatch **accumulates** and no static solution exists
6. Therefore the system **must oscillate** — neural rhythms are the sound of self-observation at finite resolution
7. The same geometric interference naturally functions as **holographic memory**

---

## Running the Code

### Interactive Demo
Open `demos/deerskin_interactive.html` in a browser. No dependencies required.

### MoiréNet Benchmark
```bash
cd benchmark
python benchmark_xor.py
```
Requires: `numpy`

### Janus Brain (Holographic Memory)
```bash
pip install gradio opencv-python numpy
python janus/janus_brainv2.py
```

### PerceptionLab Workflows
The `.json` workflows require [PerceptionLab](https://github.com/anttiluode/perceptionlab) to run. The node `.py` files go in the `nodes/` folder.

---

## Key References

- **Mannion & Kenyon (2024)** — *Artificial Dendritic Computation* — Framework for dendrites as computational elements
- **London & Häusser (2005)** — *Dendritic Computation* — Foundational review of within-neuron processing
- **Gidon et al. (2020)** — Human dendrites perform XOR-like logic in single branches
- **Lin et al. (2018)** — *All-optical machine learning using diffractive deep neural networks* — Physical proof that wave interference computes
- **Poirazi & Papoutsi (2020)** — Computational models of dendritic function
- **Pribram (1971)** — *Languages of the Brain* — Original holographic brain hypothesis
- **Raj et al. (2020)** — Spectral graph theory of brain oscillations / connectome eigenmodes

---

## Author

**Antti Luode** — Independent researcher. Builder of PerceptionLab. Has partial right temporal lobe resection, which provides unique perspective on what happens when part of the "sampling grid" is removed. Not a neuroscientist. Not claiming to be one. Building tools and sharing observations.

---

## License

MIT. Use freely. If this sparks ideas, build on them.

---

*"I really wish I had a brain upgrade. I think I would disappear completely into this stuff and think about nothing else. But here I am, completely lost in science, being very unscientific."* — from the video transcript
