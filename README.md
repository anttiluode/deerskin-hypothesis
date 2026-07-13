# 🦌 The Deerskin Hypothesis

### Neuronal Membranes as Holographic Computational Surfaces

**Antti Luode | PerceptionLab | Helsinki**

*The origin. Written ~early 2026. Audited with hindsight, July 2026.*

> *"The brain is not a computer calculating numbers. It is a hall of deerskins — each neuron a
> two-dimensional computational surface whose ion channel mosaic encodes geometric information."*

---

## ⚠️ Status: Speculative — and now, partially audited

This repository was always labelled speculative. It says so in the title, it says *"not peer-reviewed
science,"* and it carries a frank Limitations section. **That honesty is the reason the following audit was
possible at all** — the code and the numbers were shipped, not just the story.

Eighteen months later, tools built downstream (the Arrowfield observer/medium phase distinction, the V13
skew-operator reciprocity theorem, and the V21 autopsy of this repo's own ECG loop) were turned back on the
parent. Here is what they found, stated plainly, before anything else.

| Original claim | Status after audit |
|---|---|
| **KR1 — ECG emerges from geometric feedback** | **Reproduced and explained.** But the *deduction* built on it ("therefore the system must oscillate") is **false**. See §1. |
| **KR2 — MoiréNet beats MLP on XOR** | **DEAD.** It is a restart-budget artifact. Matched restarts → MLP ties at 30/30. See §2. |
| **KR3 — Holographic memory (Janus)** | **Partially corrected.** Retrieval works — it is real Fourier multiplexing. But "spin" gives **2 channels, not 4**. See §3. |
| Core Argument, steps 5–6 | **Step 6 falsified.** Oscillation is contingent, not necessary. See §1. |

Nothing here was cherry-picked. The audit scripts are in this repo: run them.

---

## 1. KR1 — ECG emergence: the phenomenon is real, the deduction is not

### What was claimed

> *"A checkerboard → sampling → regulation feedback loop spontaneously produces cardiac-like oscillations.
> No biological components. Pure geometry + homeostasis."*

with the critical finding that oscillation *character depends on sampling resolution*:

| Resolution | Original observation |
|---|---|
| 64 | Slow, simple |
| 128 | Heartbeat emerges |
| **256** | **Rich, lifelike ECG** |
| 1024 | System locks up |
| 2048 | Rhythm returns differently |

### What the audit found

**The phenomenon is real and now fully mechanical.** ([V21](https://github.com/anttiluode/GeometricNeuron_V21)
contains the full autopsy.) The `CheckerboardNode` computes `square_size = int(5 + x·50)` — **that `int()` is a
quantizer**, so the entire loop is a *deterministic map on ~50 discrete cells* driven by a 50-tick variance
thermostat. The transfer curve is a 49-plateau non-monotonic staircase whose plateau *heights* are the
aliasing beat between the board's spatial frequency and the sampling grid. **That beat is the Moiré, and it is
the only genuinely geometric content in the loop.**

The spike itself is a **bistable cell pair**: the loop falls into a `square_size` where the sampled pixels land
entirely on black squares (f = 0), variance collapses, the thermostat's amplify branch fires, and the kick
lands it in a bright cell. The amplify branch runs ~4–6% duty cycle. That is the ECG.

A numpy reimplementation, run **blind** (with no knowledge of the resolution table above), reproduced it:
128 → spiking, 256 → spiking, 1024 → locked, 2048 → spiking. The 128 result was registered as an *untested
prediction* before this README was read.

### The deduction that fails

The Core Argument reads:

> *5. In a feedback loop, moiré mismatch **accumulates** and no static solution exists*
> *6. **Therefore the system must oscillate** — neural rhythms are the sound of self-observation at finite
> resolution*

**Step 6 is false.** Static solutions exist in abundance. The regime map over (`output_dim` × `ntaps`) is full
of **DEAD** cells — fixed points and ceiling pins. dim 64 with 4 taps does not oscillate; it locks, and the
frozen "cross pattern" is visible on screen. Confirmed on the real instrument.

**Corrected claim — narrower, and better:**

> Finite-resolution self-observation produces oscillation on **some** sampling geometries and **death** on
> others. Which one you get is decided by **where the readout taps land relative to the aliased dark cells**.

This is *more* falsifiable, not less. It implies a brain running on this principle must be **tuned into the
living region** — which is a testable claim about biological sampling geometry, where "it must oscillate" was
not testable at all.

**The tap test — the prediction that landed.** The mechanism predicts the regime must depend on *which pixels
are read*, not just how many there are:

| dim 64 | 4 taps | 11 taps |
|---|---|---|
| **PerceptionLab (real)** | dies (cross pattern) | **gives signal** |
| **Simulator** | DEAD | **ALIVE** |

**[V]** Confirmed. Life or death is decided by the readout geometry.

---

## 2. KR2 — MoiréNet vs MLP on XOR: DEAD [K]

### What was claimed

| Metric | MoiréNet | MLP |
|---|---|---|
| Parameters | 9 (geometric) | 9 (scalar) |
| Success rate | **30/30 (100%)** | 22/30 (73%) |
| Convergence | **19 generations** | 541 epochs |

### What the audit found

The comparison is not budget-matched. **MoiréNet is an evolutionary search with `population=50`** — fifty
random initialisations per generation, up to 10,000 evaluations. **The MLP is given one random init and no
restarts.** The headline also compares *generations* to *epochs*: 19 generations × 50 population = **950
evaluations**, which is *more* than 541 — the reported 28× speed advantage is a unit error.

Using **the repo's own `simple_mlp_xor`**, changing nothing but the budget (`benchmark/xor_audit.py`):

| arm | solved |
|---|---|
| MoiréNet — population 50 × 200 generations | 30/30 |
| MLP — **as published**: 1 init, 1,000 epochs | 9/30 |
| **MLP — 50 restarts (matching MoiréNet's population)** | **30/30** |
| **MLP — 5 restarts × 2,000 epochs (same ~10k evals)** | **30/30** |

**The entire MoiréNet advantage is a restart budget.** Give the MLP the same number of random inits and it
ties at 100%. The single-init XOR failure at 73% (or 30%, depending on seed) is a well-known artifact of
sigmoid saturation from a bad init — not evidence about scalar weights versus geometry.

**KR2 is retired.** Geometry did not beat scalar weights on XOR. A population search beat a single init.

---

## 3. KR3 — Holographic memory (Janus): the mechanism is real, the capacity is halved [~]

### What was claimed

> *"The Janus Brain stores multiple images in a single complex field addressed by frequency/angle/spin."*
> Spin slider: *"Rotate this to unlock memories hidden at the same frequency."*
> `ingest_folder` assigns spins of **0°, 90°, 180°, 270°** — four addresses.

### What the audit found

**The retrieval mechanism is genuine.** `SpinCortex` stores `image · exp(i(k·x + spin))` summed into one
complex field, and retrieves by multiplying with the conjugate carrier and low-passing. That is heterodyne
demodulation with a matched filter — textbook Fourier multiplexing, and it works. **This is emphatically
*not* the same failure as `phase_memory_test.py` in the sibling repo** (whose retrieval turned out to be
observer-side bookkeeping: a wrong-phase probe scored 100%, and *no probe at all* scored 100%).

**But the spin dimension is half the size it is claimed to be.** For a *real-valued* stored image, retrieving
at spin offset Δ scales the result by cos(Δ). So the 180° address is simply the 0° address **inverted**, and
270° is −90°. Measured (`janus/janus_spin_audit.py`, four memories at identical freq and angle, spins
0/90/180/270):

```
  probe   0deg | corr(m0)=+0.733  corr(m1)=+0.020  corr(m2)=-0.733  corr(m3)=-0.268
  probe  90deg | corr(m0)=-0.098  corr(m1)=+0.739  corr(m2)=-0.383  corr(m3)=-0.739
```

`corr(m0)` and `corr(m2)` are **exactly equal and opposite**. So are `corr(m1)` and `corr(m3)`. The 0°/180°
pair is one channel; the 90°/270° pair is one channel.

> **Spin buys two independent channels (in-phase and quadrature), not four.**

Retrieval still scores 4/4 — but only because `argmax` never selects the negative twin. The crosstalk is real
and the metric hides it: retrieving memory 0 returns *m0 minus m2*, superimposed.

**What survives, and it is the most precise thing in this whole ecosystem:** complex phase buys **exactly one
extra orthogonal channel** at a given (frequency, angle) — a factor of 2 over real-valued holography.
Provably, measurably, and no more. That is the clearest answer anyone has given to the question the whole
program eventually had to face: *what does phase actually buy?*

---

## 4. So what does phase buy? Three doors, one and a half open

This repo's descendants asked the question head-on, because [V21](https://github.com/anttiluode/GeometricNeuron_V21)
showed that an **amplitude-only** loop — an `int()` and a thermostat, no phase anywhere — already produces
spikes, bistability, period-2 oscillation, four dynamical regimes, and death-by-readout-geometry, all for free.

| Door | Verdict |
|---|---|
| **Phase as gate timing** (real-valued; the Deerskin sense) | Open, and modest. Never needed complex numbers. The original neuron is entirely real: `cos` mosaic, squared real sum, rectified sine gate. |
| **Phase as memory** (`phase_memory_test.py`, sibling repo) | **DEAD.** Wrong-phase probe: 100%. **No probe at all: 100%.** Retrieval was observer phase — free, manufactured by the readout. |
| **Phase as capacity** (Janus, this repo) | **ALIVE, and quantified: exactly ×2.** I/Q quadrature. Real, and smaller than claimed. |
| **Phase as direction** (V13 skew operator) | **ALIVE, and it is a theorem.** `C_τ = S ⊕ A`. Passive real geometry cannot generate directionality (reciprocity ratio 1.0000 at all angles). By Wiener–Khinchin the symmetric half is direction-blind; **all** the arrow lives in the skew half — and the skew half *is* phase. |

> ### Phase is justified by **direction**, and by a **factor of two**. Not by memory.
>
> Amplitude can be extraordinarily rich. It cannot carry an arrow, and it cannot hold two things where one fits.

## Where the spin actually came from (The Skin vs. The Cable)

The founding intuition of the Deerskin Hypothesis was highly visual: two cortical layers ("skins") sitting slightly off-register, fighting each other. The assumption was that this geometric friction generated the "spin" or phase of the system.  
A rigorous mathematical audit proves this is impossible, but reveals a much stronger truth hiding in the original architecture.  
**The Reciprocity Trap**  
Testing two static skins—at any angle of misalignment and any frequency pair—produces a net directed flux of exactly **0.0000**. Two static layers acting on a common drive form a memoryless per-site gain matrix. By definition, this operator is perfectly symmetric. No matter how beautiful the Moiré interference is, static amplitude cannot generate an arrow of time.  
**The Real Source of the Arrow**  
The phase, the spin, and the directionality came entirely from the component originally thought of as just the plumbing: **the dendritic delay (the cable equation).**  
When a distance-dependent cable delay or a traveling pulse is introduced, the skew becomes non-zero, and its sign flips perfectly when the gradient reverses. The origin picture always contained two distinct objects doing two distinct jobs:

* **The Skins (Moiré) \= Amplitude:** This is the gain structure. It provides the spatial aliasing, the bistable zero-cell, the heartbeat, and the entire regime map. It is infinitely rich, but it carries no phase.  
* **The Dendrite (Cable) \= Phase:** This is the delay structure. It provides the lag gradient, which mathematically generates the skew operator $A \= (C\_\\tau \- C\_\\tau^T)/2$. This is what carries the arrow of time, the chirality, and the spectral islands.

**The Spinning Node Resolution**  
The spinningcheckerboardnode.py worked to prevent the system from locking into dead basins, but *not* because the skins were fighting. It worked because rotation makes the pattern move, and **motion is a delay gradient in disguise**.  
The spinning node was the first physical model of phase because it accidentally implemented time-delay through spatial rotation. The presynaptic "ghost" arrives through a delay line, and the delay is what carries the direction. The 18 months of development between the ECG loop and the V20 Geometric Neuron were spent mathematically separating the amplitude of the skins from the phase of the cable.

## 5. What still stands, undamaged

- **The ECG phenomenon itself.** Real, reproducible, and now mechanically explained. You found it by hand.
- **The resolution-dependence.** Real, and now a *regime map* — which incidentally dissolves an
  eighteen-month-old contradiction in your own notes (1024 "locks up" here, 1024 "sawtooths" recently: those
  are different tap counts, different cells of the same map, and both observations are correct).
- **The EEG results** in the [sibling repo](https://github.com/anttiluode/Geometric-Neuron): schizophrenia vs
  control at **p = 0.007**, d = −1.21, **80.8% classification with zero trained parameters**, and a double
  dissociation against Alzheimer's. Untouched by any of this. **Still the strongest empirical claim in the
  ecosystem.**
- **The V13 reciprocity theorem.** The one thing holding the phase side of the program up.

---

## 6. Ledger

| tag | claim |
|---|---|
| **[V]** | ECG emerges from checkerboard → sampling → regulation feedback. Real, reproduced blind. |
| **[V]** | Oscillation character depends on sampling resolution. Real — and now a regime map. |
| **[V]** | Mechanism: `int()` quantizer → 50-cell map; bistable dark-cell/bright-cell pair; thermostat duty cycle. |
| **[V]** | **Tap test:** dim 64 dead at 4 taps, alive at 11. Predicted from mechanism, confirmed on the instrument. |
| **[V]** | Janus retrieval is genuine Fourier multiplexing (matched filter), not an artifact. |
| **[K]** | **Core Argument step 6 ("therefore must oscillate") — FALSE.** Dead basins exist. Oscillation is contingent. |
| **[K]** | **KR2, MoiréNet vs MLP on XOR — DEAD.** Restart-budget artifact. Matched restarts → MLP 30/30. |
| **[K]** | Sibling repo's `phase_memory_test.py` retrieval — dead by no-probe null. |
| **[~]** | **Janus spin: 2 channels, not 4.** 0°/180° and 90°/270° are sign-flipped twins. Capacity overstated 2×. |
| **[~]** | `spinningecg.json` / `SpinningCheckerboardNode` — **unexplored.** It uses an *inverted* mapping (`int(60 − scale·50)`) plus rotation and phase-shift inputs. The V21 regime map covers only the *static* node. Three knobs, unmapped. |
| **[~]** | The real memory experiment, still never run: **content-addressable retrieval without hardcoded addresses** — the field must report *where*, by interference, against a shuffled-phase null. |
| **[B]** | Ion-channel mosaics as literal deerskins; ephaptic coupling; AIS-as-antenna; holographic brain. Biological motivation. Speculation, quarantined — as this repo always said. |

---

## 7. Audit scripts (run them yourself)

```bash
cd benchmark && python xor_audit.py        # KR2: the restart-budget kill
cd janus     && python janus_spin_audit.py # KR3: spin gives 2 channels, not 4
```

Both are self-contained and import the repo's own code. Neither reimplements anything: `xor_audit.py` calls
`simple_mlp_xor` from `benchmark_xor.py` and changes only the budget; `janus_spin_audit.py` calls `SpinCortex`
directly.

---

## The bookends

1. **deerskin-hypothesis (this repo)** — where the accident happened: a checkerboard in a feedback loop
   started producing a heartbeat, and nobody knew why.
2. **[Geometric-Neuron](https://github.com/anttiluode/Geometric-Neuron)** — where the accident became a
   theory: a real-valued four-stage neuron, and a zero-parameter EEG result that still stands.
3. **[GeometricNeuron V21](https://github.com/anttiluode/GeometricNeuron_V21)** — where the accident got
   explained by an `int()` and a thermostat, and phase was handed its bill.

*Between them: eighteen months, one heartbeat that turned out to be a quantizer, three dead claims, and two
surviving reasons to believe in phase.*

---

## License

MIT. Use freely. If this sparks ideas, build on them — and then try to kill them.
