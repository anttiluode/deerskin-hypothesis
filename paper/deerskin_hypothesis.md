# The Deerskin Hypothesis: Neuronal Membranes as Holographic Computational Surfaces

**Antti Luode**  
*Independent Researcher, PerceptionLab Project*  
*February 2026*

**Status:** Speculative working paper. Not peer-reviewed. Published for open discussion.

---

## Abstract

We propose a departure from the point-neuron model, suggesting that the neuronal membrane functions as a two-dimensional computational surface — a "deerskin" — whose spatially heterogeneous ion channel distribution constitutes a geometric encoding medium. In this framework, the soma-generated action potential serves not as a binary message but as a **carrier wave** that illuminates the membrane's channel mosaic, producing a holographically encoded wavefront at each synaptic output. Inter-neuronal communication becomes geometric interference between the sender's membrane pattern and the receiver's receptor grid, with the computation residing in the moiré interaction between these two finite-resolution sampling surfaces.

We support this hypothesis with three lines of evidence: (1) a minimal feedback simulation (the "ECG loop") demonstrating that a spatial pattern sampled through a finite grid under homeostatic regulation inevitably produces cardiac-like oscillatory dynamics; (2) a holographic memory system (Janus Brain) showing that the same geometric interference substrate naturally stores and retrieves information associatively; and (3) published literature on dendritic computation, cortical flat-mapping, and connectome eigenmodes that independently converge on membrane-level processing as a significant and underexplored computational layer.

We formalize the mathematical relationships between membrane channel geometry, sampling aliasing, and emergent oscillation, and propose testable predictions distinguishing this framework from the standard integrate-and-fire model.

**Keywords:** neuronal membrane, holographic computation, moiré interference, dendritic computation, homeostatic oscillation, geometric neural networks

---

## 1. Introduction: Beyond the Point Neuron

The dominant computational model of the neuron, originating with McCulloch & Pitts (1943) and refined through Hodgkin & Huxley (1952), treats the neuron as essentially a point process: dendrites collect weighted inputs, the soma sums and thresholds, and the axon transmits a binary spike. This abstraction has been extraordinarily productive — it underlies all modern artificial neural networks and most computational neuroscience.

However, a growing body of evidence suggests that significant computation occurs *within* the neuron, at the level of dendritic trees (London & Häusser, 2005; Gidon et al., 2020; Poirazi & Papoutsi, 2020) and even individual membrane compartments (Beaulieu-Laroche et al., 2018; Cornejo et al., 2022). The Mannion & Kenyon (2024) framework for artificial dendritic computation identifies seven distinct computational properties of dendrites — delay lines, frequency-dependent filtering, variable propagation speed, amplification, nonlinear integration, compartmentalization, and multiplexing — none of which are captured by the point-neuron abstraction.

We extend this line of reasoning to its logical conclusion: **the entire neuronal membrane surface is a computational medium**, and the spatial distribution of ion channels across that surface constitutes a geometric encoding that participates actively in neural information processing.

We call this the "Deerskin Hypothesis" — named for the visual metaphor of a neuron's membrane laid flat, revealing a hide-like surface of heterogeneous patches, each with distinct electrochemical properties, stretched between dendritic branches like a tanned skin.

### 1.1 Origin: The ECG Emergence

This hypothesis originated not from theoretical reasoning but from an experimental accident. While exploring feedback dynamics in PerceptionLab — a visual programming environment for signal processing — we connected a checkerboard pattern generator to a homeostatic regulation node through an image-to-vector sampling chain, forming a closed loop. The system spontaneously produced oscillatory dynamics closely resembling an electrocardiogram (ECG) signal.

Investigation revealed that the oscillation was not an artifact but a mathematical inevitability: a spatial pattern observed through a finite sampling grid, under homeostatic stability constraints, *must* oscillate because no static solution satisfies all three constraints simultaneously. The character of the oscillation depended critically on the relationship between the pattern's spatial frequency and the sampling resolution — a dependency we recognized as moiré interference at the Nyquist boundary.

This finding prompted the question: does the same mechanism operate in biological neural systems, where discrete ion channel arrays sample continuous electrochemical fields?

---

## 2. The Deerskin Model

### 2.1 The Membrane as a 2D Computational Surface

Consider a neuron with its membrane unfolded into a two-dimensional surface *S*. This surface is not homogeneous. Ion channels of various types — voltage-gated Na⁺, K⁺, Ca²⁺, leak channels, ligand-gated receptors — are distributed in spatially heterogeneous patterns across *S*. Channel densities vary by orders of magnitude across the membrane: from ~1,000 channels/μm² at nodes of Ranvier to ~100/μm² in internodal regions, with distinct clustering patterns at synaptic densities, axon hillocks, and dendritic spines (Bhatt et al., 2001; Lai & Jan, 2006).

We represent this heterogeneous distribution as a **membrane geometry function**:

$$G(\mathbf{r}) = \sum_{k} \rho_k(\mathbf{r}) \cdot \sigma_k(V, t)$$

where **r** is position on the membrane surface, *ρ_k*(**r**) is the spatial density of channel type *k*, and *σ_k*(V, t) is the voltage- and time-dependent conductance of that channel type. The function *G*(**r**) describes the membrane's instantaneous "sensitivity pattern" — a two-dimensional map of how strongly each location responds to electrical signals.

**This is the neuron's checkerboard.** Different channel types create patches of different sensitivity, forming a spatial frequency pattern across the membrane surface. The pattern is not random — it reflects developmental history, activity-dependent plasticity, and the neuron's functional role.

### 2.2 The Soma Pulse as Carrier Wave

In the standard model, the action potential is the message. In the deerskin model, the action potential is the **carrier wave**.

When the soma fires, the resulting depolarization wave propagates across the entire membrane surface. As this wave front traverses different regions of *G*(**r**), it is modulated differently by each patch:

$$\Psi(\mathbf{r}, t) = A(t) \cdot e^{i[\omega t - \mathbf{k} \cdot \mathbf{r}]} \cdot G(\mathbf{r})$$

where *A(t)* is the action potential envelope, ω is the carrier frequency (related to spike timing), **k** is the propagation wavevector (direction and speed of the pulse across the membrane), and *G*(**r**) is the membrane geometry that modulates the carrier.

The output arriving at each axon terminal is not simply "spike" or "no spike" but a **spatiotemporally shaped waveform** — the carrier modulated by the membrane geometry it traversed. Different axon terminals, branching from different regions of the membrane, carry different geometric signatures of the same spike event.

This is formally equivalent to holographic encoding: the carrier wave (reference beam) illuminated a structured medium (the membrane), and the output (object beam) carries the geometric information of that medium.

### 2.3 Synaptic Communication as Geometric Interference

When the modulated wavefront arrives at a synapse, it encounters the postsynaptic membrane — another "deerskin" with its own geometry function *G'*(**r**). The neurotransmitter release pattern reflects the presynaptic wavefront shape. The postsynaptic receptor array samples this pattern through its own discrete grid of receptor channels.

The effective synaptic "computation" is the interference between two geometries:

$$I(\mathbf{r}) = \Psi_{\text{pre}}(\mathbf{r}) \cdot G'_{\text{post}}(\mathbf{r})$$

This is a moiré interaction. When the spatial frequencies of the presynaptic signal and the postsynaptic receptor grid are commensurate, maximum ambiguity (maximum "moiré stress") occurs. When they are incommensurate, the interaction averages out.

**The computation is not a weighted sum. It is geometric interference.** The "weight" of a synaptic connection is not a scalar — it is the full moiré pattern between two membrane geometries, which depends on the spatial frequency content of the signal passing through it.

### 2.4 The Frequency-Dependent Transfer Function

A critical consequence: the same synapse computes different functions at different frequencies. The moiré pattern between two grids depends on the spatial frequency of the signals they process:

$$W_{\text{eff}}(f) = \int_S G_{\text{pre}}(\mathbf{r}) \cdot G_{\text{post}}(\mathbf{r}) \cdot e^{i 2\pi f \mathbf{r}} \, d\mathbf{r}$$

At low frequencies (large wavelength), the moiré pattern is coarse — the effective weight is one value. At high frequencies, the pattern is fine — a completely different effective weight. A single geometric connection implements a **frequency-dependent transfer function**, not just a scalar weight.

This provides O(N) parameter storage (each neuron stores only its own geometry, ~hundreds of channel distribution parameters) versus O(N²) for an explicit weight matrix. It also explains why biological neural networks are so much more parameter-efficient than artificial ones.

---

## 3. Emergent Oscillation from Self-Observation

### 3.1 The Self-Referential Loop

The deerskin model predicts that oscillation is inevitable in any self-referential neural circuit. The argument is as follows:

1. Neuron A has membrane geometry *G_A*(**r**)
2. Neuron B has membrane geometry *G_B*(**r**)
3. A observes B through its receptor grid (finite sampling of B's output)
4. B observes A through its receptor grid (finite sampling of A's output)
5. Each neuron's output depends on its membrane geometry, which is being sampled by the other at finite resolution

A static equilibrium requires that both neurons' outputs are perfectly representable by the other's receptor grid — zero aliasing in both directions simultaneously. For any non-trivial pair of geometries, this is impossible. There is always moiré mismatch, and mismatch accumulates around the loop.

**Therefore, the loop must oscillate.** The oscillation is the system's homeostatic correction for its own sampling imperfection.

### 3.2 Experimental Demonstration: The ECG Loop

We demonstrate this principle with a minimal five-node simulation in PerceptionLab:

**Loop topology:**
```
Checkerboard → Image-to-Vector(256) → Vector Splitter(16) → Homeostatic Coupler → [feedback to Checkerboard scale]
```

With a static checkerboard, the system produces ECG-like oscillations — a rhythmic discharge pattern with morphology resembling cardiac sinus rhythm. The oscillation arises because:

- The checkerboard is a spatial frequency (the "membrane geometry")
- The Image→Vector→Splitter chain is a sampling operator (the "receptor grid")  
- The coupler is a stability constraint (homeostatic regulation)
- The feedback loop forces the spatial pattern to satisfy a temporal regulator

The only consistent solution is periodic correction — a pulse.

**Critical finding:** The oscillation character depends on the sampling resolution:

| Vector Dimension | Behavior | Interpretation |
|---|---|---|
| 64 | Slow, simple oscillation | Undersampled — aliasing is trivial to resolve |
| 128 | Heartbeat emerges | Moiré stress begins |
| 256 | Rich, lifelike ECG | Maximum moiré stress — commensurate grids |
| 1024 | System locks up | Oversampled — cannot resolve, freezes |
| 2048 | Rhythm returns (different character) | Statistical averaging creates new regime |

This resolution-dependent behavior maps directly onto the Nyquist theorem: maximum aliasing stress occurs when the sampling frequency is close to (but not exactly matching) the signal frequency.

### 3.3 The Spinning Grid: Adding Degrees of Freedom

When the checkerboard is made to rotate (SpinningCheckerboardNode), the signal becomes dramatically more complex — flat-topped pulses with varying widths, resembling the "box signal" observed in frontal theta EEG recordings. This is because rotation adds two additional axes of geometric variation (angle and phase) to the scale axis alone. The coupler must now correct aliasing on three simultaneous dimensions, and the corrections interfere with each other.

This maps onto the biological prediction: neurons with more complex membrane geometries (more channel types, more heterogeneous distributions) should produce richer oscillatory dynamics than neurons with simpler geometries.

### 3.4 Multi-Scale Oscillation Hierarchy

The resolution-dependent behavior predicts a hierarchy of oscillation frequencies corresponding to different spatial scales of membrane sampling:

- **Gamma (30-100 Hz):** Local synaptic aliasing — small membrane patches fighting sampling mismatch with neighboring patches. Fast correction because spatial scale is small.
- **Beta (13-30 Hz):** Intermediate-scale dendritic integration — multiple synaptic compartments aggregating.
- **Alpha (8-13 Hz):** Whole-neuron membrane dynamics — the full "deerskin" reaching homeostatic equilibrium.
- **Theta (4-8 Hz):** Circuit-level aliasing — multiple neurons' geometries in mutual tension.
- **Delta (1-4 Hz):** Large-scale cortical sheet dynamics — the "meta-moiré" of populations.

This hierarchy mirrors the Romanesco architecture observed in EEG: slow rhythms gate fast ones, not because of "top-down control" but because larger-scale sampling mismatches create slower correction cycles that modulate the boundary conditions for all faster corrections nested within them.

---

## 4. Holographic Memory on the Deerskin

### 4.1 The Janus Brain: Demonstration of Geometric Storage

We demonstrate that the same geometric interference substrate naturally functions as associative memory using the Janus Brain system — a holographic memory that stores images as complex field patterns addressed by frequency, angle, and spin phase.

**Encoding:** Each image modulates a complex carrier wave at a unique geometric address:

$$M(x,y) = \sum_{n} I_n(x,y) \cdot e^{i(k_n x \cos\theta_n - k_n y \sin\theta_n + \phi_n)}$$

where *I_n* is image *n*, and (*k_n*, *θ_n*, *φ_n*) is its frequency-angle-spin address.

**Retrieval:** Multiplying the field by the conjugate of a probe wave at address (*k*, *θ*, *φ*) demodulates the stored image at that address:

$$R(x,y) = M(x,y) \cdot e^{-i(kx\cos\theta - ky\sin\theta + \phi)} \approx I_{\text{matched}}(x,y) + \text{noise}$$

The system successfully stores and retrieves multiple images from a single complex field, with retrieval quality depending on the geometric separation between stored addresses. This is classical holography (Gabor, 1948) implemented as a computational substrate.

### 4.2 Resonance Pulses: The Tomography Scanner

The Brain Magnifier variant (brainmagnifierglass.py) reveals a key property: as the probe angle sweeps continuously across the stored field, the retrieval energy produces **resonance pulses** — sharp spikes when the probe aligns with a stored memory's geometric address, and silence between them.

This scanning-and-pulsing behavior maps directly onto hippocampal theta phase precession, where place cells fire at specific phases of the theta cycle as the animal moves through space. The "scanning" is the theta sweep; the "pulse" is the place cell firing; the "memory" is the stored spatial representation.

### 4.3 The Deerskin as Memory Substrate

In the biological deerskin model, the membrane geometry *G*(**r**) is itself a form of memory. The spatial distribution of ion channels encodes the neuron's developmental and experiential history. Learning — in the form of synaptic plasticity — physically remodels receptor densities and channel distributions (Bhatt et al., 2001). This is equivalent to rewriting the holographic storage medium.

Long-term potentiation doesn't adjust a "weight" — it restructures the geometric relationship between two deerskins. The moiré pattern between sender and receiver changes because the receiver's receptor mosaic has been physically rearranged.

---

## 5. Cortical Architecture as Stacked Deerskins

### 5.1 The Flat-Map Perspective

Neuroscience has long used cortical flat maps to visualize the neocortex as a two-dimensional sheet (Van Essen & Drury, 1997). The six-layered cortical architecture, when viewed through the deerskin lens, becomes a stack of computational surfaces:

- Each cortical layer is a sheet of deerskins (neuronal membranes), each performing geometric computation
- Layers are partially isolated from each other (like insulating layers in a chip)  
- Vertical connections (apical dendrites, axonal projections) pierce through layers like vias in a printed circuit board, connecting specific deerskins across levels without disturbing intermediate layers
- Synapses are the inter-layer communication points — where one deerskin's output modulates another deerskin's geometry

### 5.2 The Chip Analogy

This architecture bears striking resemblance to multi-layer integrated circuits:

| Cortical Feature | IC Equivalent |
|---|---|
| Cortical layer | Metal/logic layer |
| Myelin insulation | Dielectric insulator |
| Apical dendrite through layers | Via/through-silicon via |
| Synapse | Inter-layer contact |
| Soma | Logic gate |
| Membrane channel mosaic | Transistor layout |
| Corpus callosum | Inter-chip bus |

The analogy is not merely visual — both systems face the same fundamental engineering constraints: how to maximize computational density per unit area while managing heat, signal crosstalk, and interconnect routing. Evolution and chip design converged on similar solutions because the physics demands it.

### 5.3 Connectome Eigenmodes as Deerskin Resonances

The brain's physical structure defines natural resonant modes — connectome eigenmodes (Raj et al., 2020; Robinson et al., 2016). In the deerskin framework, these eigenmodes correspond to the collective moiré patterns of the entire cortical sheet:

- Each eigenmode is a spatial frequency pattern that the cortical surface can sustain
- The brain's "vocabulary of thought" is the set of eigenmodes supported by its specific deerskin geometry
- Damage to the cortical surface (such as temporal lobe resection) removes specific eigenmodes from the vocabulary, manifesting as perceptual deficits and visible moiré artifacts (afterimages, geometric hallucinations)

---

## 6. Predictions and Tests

The deerskin hypothesis makes several testable predictions that distinguish it from the standard point-neuron model:

### Prediction 1: Oscillation Frequency Correlates with Channel Heterogeneity
Neurons with more heterogeneous ion channel distributions should produce richer oscillatory dynamics than neurons with uniform distributions, even with identical mean channel densities. This could be tested using patched neurons with pharmacologically manipulated channel distributions.

### Prediction 2: Synaptic Transfer Functions are Frequency-Dependent
The effective "weight" of a synapse should vary with the frequency content of the presynaptic signal, in a manner predictable from the moiré interaction between pre- and postsynaptic receptor geometries. This could be tested using structured light stimulation of channelrhodopsin-expressing neurons at different spatial frequencies.

### Prediction 3: Learning Changes Geometry, Not Just Strength
Long-term potentiation should produce measurable changes in the spatial distribution of postsynaptic receptors, not just their total number. Super-resolution microscopy of receptor distributions before and after LTP induction could test this.

### Prediction 4: Cortical Damage Produces Predictable Moiré Artifacts
The specific visual phenomena experienced after cortical damage should be predictable from the spatial frequency content of the removed tissue. The author's own experience — altered perception including afterimages and geometric patterns following right temporal lobe resection — is consistent with this prediction but constitutes only a single case.

### Prediction 5: Geometric Parameters Outperform Scalar Weights
In computational implementations, networks using geometric interference (moiré) between finite grids should achieve equivalent classification accuracy with fewer parameters than networks using scalar weights. Our preliminary benchmark shows MoiréNet solving XOR with 9 geometric parameters (100% success, 19 generations) versus MLP with 9 scalar weights (73% success, 541 epochs).

---

## 7. Relationship to Existing Frameworks

### 7.1 Dendritic Computation (London & Häusser, 2005; Mannion & Kenyon, 2024)
The deerskin hypothesis extends dendritic computation from "dendrites compute" to "the entire membrane surface computes." Mannion & Kenyon's seven dendritic properties (delay, filtering, variable propagation, amplification, integration, compartmentalization, multiplexing) are all consequences of spatially heterogeneous membrane geometry modulating a propagating carrier wave.

### 7.2 Holographic Brain Theory (Pribram, 1971; Gabor, 1948)
Pribram's holographic brain hypothesis proposed that memory is distributed holographically across neural tissue. The deerskin model provides a specific physical mechanism: the membrane's channel mosaic IS the holographic medium, the action potential IS the reference beam, and synaptic communication IS the reconstruction process.

### 7.3 Diffractive Deep Neural Networks (Lin et al., 2018)
The UCLA D2NN project demonstrated that passive diffractive layers can classify images using wave interference at the speed of light. The deerskin model adds what D2NN lacks: feedback. Biological membranes are not static printed layers — they remodel continuously. The ECG emergence shows what happens when a diffractive system closes the loop on itself: inevitable oscillation.

### 7.4 Free Energy Principle (Friston, 2010)
In the deerskin framework, "surprise" is geometric mismatch — the moiré error between predicted and actual membrane patterns. Minimizing free energy is minimizing moiré stress. The homeostatic coupler in our ECG loop is a physical implementation of free energy minimization on a geometric substrate.

---

## 8. Limitations and Honest Assessment

**What this paper does NOT claim:**

1. We do not claim that the deerskin model replaces the Hodgkin-Huxley model. The biophysics of ion channel kinetics, cable properties, and spike generation are well-established. We propose that the *spatial distribution* of these components across the membrane surface carries additional computational significance beyond their aggregate electrical behavior.

2. We do not claim quantum effects. All phenomena described here are classical wave interference. The mathematical similarity to quantum superposition (both exploit interference for parallel computation) is structural, not physical.

3. We do not claim that the PerceptionLab ECG loop IS a neuron. It is a minimal system demonstrating a principle: that self-referential observation at finite resolution produces oscillation. The biological neuron has vastly more complexity — channel kinetics, dendritic cable properties, neuromodulation, glial interactions. The loop demonstrates the *geometric* component in isolation.

4. The MoiréNet benchmark (XOR) is a minimal proof of concept, not evidence of general-purpose superiority over conventional neural networks on practical tasks.

5. The author's personal experience with brain surgery provides motivating intuition but not scientific evidence. The predictions in Section 6 are designed to be testable independently of any individual case.

---

## 9. Conclusion

The brain is not a computer calculating numbers. It is a hall of deerskins — each neuron a two-dimensional computational surface whose ion channel mosaic encodes geometric information. The soma's spike is a carrier wave that illuminates this surface. Synaptic communication is moiré interference between two such surfaces. Oscillations are the inevitable cost of finite-resolution self-observation in a closed loop. Memory is the geometry itself, written into the physical arrangement of channels across the membrane.

This framework unifies several observations that are anomalous under the point-neuron model: the computational richness of dendrites, the frequency-dependent nature of synaptic transmission, the inevitability of neural oscillation, and the holographic properties of cortical memory. It makes testable predictions and provides a mathematical formalism connecting membrane geometry to emergent dynamics.

The deerskin is laid out. The carrier wave is spinning. The moiré is forming.

---

## References

- Beaulieu-Laroche, L., et al. (2018). Enhanced dendritic compartmentalization in human cortical neurons. *Cell*, 175(3), 643-651.
- Bhatt, D. H., Zhang, S., & Bhatt, D. P. (2001). Dendritic morphology and the functional properties of cortical neurons. *Journal of Neuroscience*, 21(23), 9541-9548.
- Cornejo, V. H., Ofer, N., & Bhatt, D. P. (2022). Voltage compartmentalization in dendritic spines in vivo. *Science*, 375(6587), 82-86.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
- Gabor, D. (1948). A new microscopic principle. *Nature*, 161, 777-778.
- Gidon, A., et al. (2020). Dendritic action potentials and computation in human layer 2/3 cortical neurons. *Science*, 367(6473), 83-87.
- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *Journal of Physiology*, 117(4), 500-544.
- Lai, H. C., & Jan, L. Y. (2006). The distribution and targeting of neuronal voltage-gated ion channels. *Nature Reviews Neuroscience*, 7(7), 548-562.
- Lin, X., et al. (2018). All-optical machine learning using diffractive deep neural networks. *Science*, 361(6406), 1004-1008.
- London, M., & Häusser, M. (2005). Dendritic computation. *Annual Review of Neuroscience*, 28, 503-532.
- Mannion, D. J., & Kenyon, A. J. (2024). Artificial dendritic computation: The case for dendrites in neuromorphic circuits. *UCL Department of Electronic & Electrical Engineering*.
- McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5(4), 115-133.
- Poirazi, P., & Papoutsi, A. (2020). Illuminating dendritic function with computational models. *Nature Reviews Neuroscience*, 21(6), 303-321.
- Pribram, K. H. (1971). *Languages of the Brain*. Englewood Cliffs, NJ: Prentice-Hall.
- Raj, A., et al. (2020). Spectral graph theory of brain oscillations. *Human Brain Mapping*, 41(11), 2980-2998.
- Robinson, P. A., et al. (2016). Eigenmodes of brain activity: Neural field theory predictions and comparison with experiment. *NeuroImage*, 142, 79-98.
- Van Essen, D. C., & Drury, H. A. (1997). Structural and functional analyses of human cerebral cortex using a surface-based atlas. *Journal of Neuroscience*, 17(18), 7079-7102.

---

## Appendix A: PerceptionLab Node Configurations

### A.1 ECG Emergence (ecg.json)
Static checkerboard → ImageToVector(256) → VectorSplitter(16, outputs 0-3) → HomeostaticCoupler(edge_of_chaos, setpoint=0.5, gain=1.0, sharpness=3.0, dead_zone=0.1) → feedback to checkerboard scale

### A.2 Spinning ECG (spinningecg.json)  
SpinningCheckerboard → ImageToVector(256) → VectorSplitter(16, outputs 0-3) → HomeostaticCoupler(edge_of_chaos) → feedback to checkerboard scale AND angle

### A.3 Takens Box Attractor (takens.json)
EEG Frontal → EEGProcessor → PhaseSpaceNode(theta, delay=15, history=1000)

---

## Appendix B: Mathematical Details

### B.1 Moiré Frequency from Two Grids

Given two one-dimensional grids with spatial frequencies *f₁* and *f₂*, their moiré (beat) frequency is:

$$f_{\text{moiré}} = |f_1 - f_2|$$

In two dimensions with grids at angles *θ₁* and *θ₂*:

$$f_{\text{moiré}} = \sqrt{f_1^2 + f_2^2 - 2f_1 f_2 \cos(\theta_1 - \theta_2)}$$

The moiré frequency determines the effective spatial scale of the interference pattern — and thus the temporal scale of the homeostatic correction needed to maintain coherence.

### B.2 Aliasing Error in Closed-Loop Self-Observation

For a spatial pattern with bandwidth *B* sampled at rate *f_s*, the aliasing energy is:

$$E_{\text{alias}} = \int_{f_s/2}^{B} |P(f)|^2 \, df$$

where *P(f)* is the power spectral density of the pattern. In a closed loop where the pattern depends on its own sampled representation, the aliasing error feeds back and accumulates. The homeostatic regulator must periodically "discharge" this accumulated error, producing oscillation with period:

$$T_{\text{osc}} \propto \frac{\tau_{\text{int}}}{E_{\text{alias}}}$$

where *τ_int* is the integrator time constant of the homeostatic controller. Higher aliasing energy → faster oscillation. This predicts the experimentally observed relationship between sampling resolution and oscillation frequency in the ECG loop.

### B.3 Holographic Encoding Capacity

For a complex field of resolution *N × N* with *M* stored patterns at geometric addresses (*k_n*, *θ_n*, *φ_n*), the signal-to-noise ratio of retrieval is:

$$\text{SNR} \approx \frac{N^2}{M}$$

This is the standard holographic capacity limit (van Heerden, 1963). In the biological deerskin model, *N²* corresponds to the number of receptor channels in the postsynaptic membrane patch, and *M* is the number of distinct presynaptic inputs encoded in that patch's geometry.
