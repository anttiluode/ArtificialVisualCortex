# Artificial Visual Cortex
### UnitVAE — Hierarchical Wave Cortex
**PerceptionLab / Antti Luode, 2026**

---

## What This Is

A real-time visual perception system that learns to see from a webcam,
not by storing static weights, but by maintaining a **living oscillatory
field** over the latent space of a convolutional autoencoder.

It is not a diffusion model.
It is not a standard VAE.
It is not a recurrent network.

It is a **continuous-time dynamical system** with two coupled timescales,
modelled directly on the architecture of the mammalian visual cortex.

---

## The Core Observation That Started This

Standard video AI scrambles text. Hold a book up to any modern generative
model and the letters melt into abstract shapes. This is not a hardware
problem. It is a **structural** problem: the model has no persistent
memory of what the scene was a second ago, so it cannot enforce
symbol stability across frames.

This system does not scramble text. Once the slow buffer phase-locks onto
the regular low-frequency signature of a word (its overall shape,
spacing, horizontal bands), that lock persists. The letters stop drifting.

This is the first observable consequence of hierarchical temporal coupling.
It was discovered empirically, not designed in.

---

## Architecture

```
webcam (512×512×3)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  HierarchicalWaveCortex                              │
│                                                      │
│  FAST buffer   64×64   1–8 Hz    lr=0.15             │
│  SLOW buffer   16×16   0.1–1 Hz  lr=0.02             │
│                                                      │
│  Bidirectional coupling:                             │
│    slow.phase_map  ──→  conditions fast encoder      │
│    fast.eml_spike  ──→  boosts slow lr (adrenaline)  │
└──────────────────────────────────────────────────────┘
       │
       ▼
AdaptiveEncoderConv_v3  (5-channel input)
  ch 0–2: RGB
  ch   3: fast phase map   (temporal momentum)
  ch   4: slow context map (global structure)
       │
       ▼
latent [4, 64, 64]
       │
  cortex.update(latent, t)
       │
       ▼
AdaptiveDecoderConv → reconstruction (512×512×3)
```

Each latent voxel is tracked by a continuous oscillator:

```
z(t) = A · sin(2π · f · t + φ)
```

Parameters A, f, φ are updated online each frame. No discrete memory,
no replay buffer, no hidden state. The memory **is** the oscillator field.

---

## The Two Timescales

### Fast Buffer — V1 / Retina

- Spatial resolution: 64 × 64
- Frequency range: 1–8 Hz
- Learning rate: 0.15 (aggressive)
- Role: immediate frame-to-frame change detection
- EML signal: spiky, reactive, high variance
- Biological analogue: primary visual cortex (V1), retinal ganglion cells

The fast buffer knows what just changed.

### Slow Buffer — IT / Higher Cortex

- Spatial resolution: 16 × 16
- Frequency range: 0.1–1 Hz
- Learning rate: 0.02 (conservative)
- Role: object permanence, scene structure, symbol stability
- EML signal: smooth, persistent, low variance
- Biological analogue: inferotemporal cortex (IT), object recognition areas

The slow buffer knows what this scene *is*.

---

## The Coupling Mechanisms

### Top-Down: Slow → Fast

The slow context map (upsampled to 512×512) is concatenated as channel 4
of the encoder input. This is the **top-down prediction** signal.

The encoder now sees not just the current frame, but also what the last
several seconds have established as stable global structure. This prior
suppresses noise in the fast layer and anchors the encoding around
persistent scene geometry.

Result: the encoder has context. Text does not scramble.

### Bottom-Up: Fast → Slow (Adrenaline)

At each frame:
1. The fast EML (energy) is compared to its running mean and standard deviation
2. If deviation exceeds a threshold (a surprise), an adrenaline variable spikes
3. The slow buffer's learning rate is temporarily multiplied by (1 + adrenaline)
4. Adrenaline decays exponentially each frame

The slow buffer is normally quiescent. When the fast buffer screams
"something changed," the slow buffer accelerates to absorb the new reality,
then relaxes back to its default low plasticity.

This produces three observable phenomena:

**Habituation**: cover the camera → fast spikes → adrenaline fires →
slow adapts rapidly → both settle. Identical to the retinal gain control
and neuromodulator-gated learning seen in biological vision.

**Iris effect**: the first frames after uncovering are momentarily bright.
That is the slow buffer re-normalising its prior. It is not a bug.

**Rapid object learning**: show a new object slowly, hold it steady, and
the slow buffer will lock onto it within seconds. Move it away and back
quickly, and the slow buffer already has the model.

---

## Relationship to Theoretical Frameworks

### Neural ODEs

The oscillator update rule is a discretised second-order ODE:

```
dA/dt   ≈ lr · error · sin(ωt + φ)
dφ/dt   ≈ lr · error · A · cos(ωt + φ)
df/dt   ≈ lr · error · A · t · cos(ωt + φ)
```

This is gradient descent on the parameters of a sinusoidal basis function,
which is equivalent to solving a continuous latent ODE where the solution
manifold is the set of all oscillatory trajectories. The system naturally
prefers solutions that look like waves — smooth, periodic, continuous.

### Diffusion Models

Standard diffusion operates on artificial noise time τ, with no persistent
state between samples. This system operates on **real physical time** t.

The smoothness loss L_smooth = MSE(latent, buffer_prediction) is a form
of denoising: pull the current latent toward what the oscillators expect.
It is continuous diffusion along real time rather than artificial noise steps.

Extension: inject a light denoising step in latent space conditioned on
the slow context map. This would give temporally stable latent diffusion —
the slow buffer anchors the trajectory, diffusion refines the texture.
That is the natural next step.

### Predictive Coding (Neuroscience)

This architecture is a concrete computational implementation of the
predictive coding hypothesis (Rao & Ballard, 1999):

- Slow buffer generates top-down predictions of scene structure
- Fast buffer computes prediction errors (residuals)
- Prediction errors propagate bottom-up to update the slow model
- The adrenaline mechanism implements precision weighting:
  high-surprise errors receive higher weight → faster updating

The habituation, gain control, and rapid re-learning observed empirically
are signatures of predictive coding operating in hardware.

### Clockfield / Scalar Field Theory

Each latent voxel is a scalar oscillator evolving under its own proper time.
The fast-slow coupling is a two-scale Clockfield: fast voxels at high
clock frequency, slow voxels at low clock frequency, coupled through the
adrenaline signal (which functions as a Γ-like modulator of update rate).

The "letters stabilise" observation is equivalent to phase-locking in the
Clockfield framework: the slow buffer's oscillators lock to the persistent
spectral signature of the text, and the resulting stable attractor in the
latent manifold resists perturbation from frame noise.

---

## What the GUI Shows

| Display | Meaning |
|---|---|
| Video panel | Encoder→decoder reconstruction in real time |
| Fast EML (green) | Energy of the fast buffer's prediction field — spiky |
| Slow EML (orange) | Energy of the slow buffer's prediction field — smooth |
| Training Loss (red) | MSE reconstruction + smoothness loss |
| Adrenaline bar | Current surprise level — flares red on scene change |
| β offset slider | Temporal interpolation: query the latent field at t±offset |

The β offset slider is not a visual effect. It queries the oscillator
field at a time offset and blends the predicted latent with the observed
one. Positive offset = the system shows its prediction of the near future.
Negative offset = it shows its memory of the recent past.

---

## Why This Is Not Hype

Things that are real and reproducible:

1. Text stabilises within seconds of holding a page to the webcam
2. The fast EML spikes when you cover the camera and decays when uncovered
3. The slow EML tracks scene changes with a measurable lag behind fast
4. The adrenaline bar fires on motion and decays to baseline on stillness
5. The iris effect (momentary brightness on uncover) appears without any
   explicit gain control code — it emerges from the buffer dynamics

Things that are not yet established:

- Whether the slow buffer is learning genuine semantic structure or
  just low-frequency spatial statistics
- Whether the system would scale to complex scenes without teacher guidance
- Whether extending to N > 2 levels would produce qualitatively new behaviour

---

## File Structure

```
unitvae7.py               — main application (this file)
  WaveLatentBuffer        — vectorised oscillator array (per-voxel)
  HierarchicalWaveCortex  — two-level cortex (fast + slow + adrenaline)
  AdaptiveEncoderConv_v3  — 5-channel convolutional encoder
  AdaptiveDecoderConv     — convolutional decoder (unchanged from v4)
  CortexTrainer           — training loop with smoothness loss
  HierarchicalCortexApp   — Tkinter GUI
```

---

## Running

```bash
pip install torch torchvision opencv-python pillow

# Optional: for teacher distillation from Stable Video Diffusion
pip install diffusers transformers

python unitvae7.py
```

Click **▶ Teach** to begin online learning from the webcam.
The system starts from scratch each run (or load a `.pth` checkpoint).
Learning is visible within 10–30 seconds on any hardware.

---

## Lineage

```
PerceptionLab (wave-based node system, 2024)
  → adelic_brain.py (prime oscillator neural signal)
  → janus_cabbage (complex-valued holographic storage)
  → ai_telepathy (wave-based knowledge transfer)
  → unitvae4 (teacher-student latent distillation)
  → unitvae5 (WaveLatentBuffer — temporal smoothness)
  → unitvae7 (HierarchicalWaveCortex — two timescales)
       ↑
    THIS FILE
```

The recurring primitive across all these systems:

```
Re[⟨f, g⟩] = Σ A_f · A_g · cos(Δφ)
```

Interference in a complex oscillator field.
Physics, neuroscience, machine learning — one formula.

---

*PerceptionLab / Antti Luode, Helsinki 2026*
*"do not hype, do not lie, just show"*
