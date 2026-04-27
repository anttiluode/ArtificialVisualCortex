"""
unitvae5.py  —  Adaptive VAE with Wave Latent Buffer
=====================================================
PerceptionLab / Antti Luode, 2026

What this adds on top of unitvae4.py:

1. WaveLatentBuffer
   A vectorized WaveNeuron layer that runs over the latent space [4, 64, 64].
   Every spatial position in every latent channel has its own oscillator (A, f, φ).
   On each frame the oscillators are updated to track the incoming latent.
   Key capability: query the buffer at any real-valued t for temporal interpolation —
   sub-frame latents, slow-motion, etc.

2. Temporal EML conditioning channel
   The EML signal of the latent buffer (|Z(t)| − arg(Z(t))) is computed every frame
   and downsampled to a [1, 64, 64] "phase map". This is injected into the encoder
   as a 4th input channel so the student knows where it is in its oscillation cycle.
   The encoder learns to anticipate motion, not just react to it.

3. Temporal smoothness loss
   An extra loss term penalises |current_latent − buffer_prediction|².
   The buffer predicts what the next latent should be based on its oscillator state.
   This trains the student to produce latents that are physically continuous
   in the wave sense, reducing frame-to-frame jitter without explicit optical flow.

4. Latent wave visualiser
   A strip chart in the GUI shows the EML "brain wave" of the latent buffer —
   the aggregate oscillation signal — updated in real time.

Architecture diagram:

    webcam frame (512×512×3)
           │
           ▼
    [EML phase map (64×64×1)] ──────────────────────────┐
    (from WaveLatentBuffer)                               │
                                                          ▼
                                          AdaptiveEncoderConv_v2
                                          (4 input channels: RGB + phase)
                                                          │
                                                          ▼
                                               latent [4, 64, 64]
                                                    │         │
                                                    │    WaveLatentBuffer
                                                    │    .update(latent, t)
                                                    │         │
                                                    │    smoothness_loss
                                                    │    = |latent − predicted|²
                                                    ▼
                                          AdaptiveDecoderConv
                                                    │
                                                    ▼
                                           reconstructed frame
"""

import os, sys, types, threading, time
import cv2, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as T
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# ─── Triton monkey-patch ──────────────────────────────────────────────────────
os.environ["DIFFUSERS_NO_IP_ADAPTER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Triton patch ─────────────────────────────────────────────────────────
# Problem: diffusers/transformers call importlib.util.find_spec("triton")
# which raises ValueError if triton is in sys.modules with __spec__=None.
# Solution: import diffusers BEFORE touching sys.modules, then patch after.
# The DummyAutotuner is only needed if triton is present but incomplete.
import importlib.util as _iutil

_triton_spec = _iutil.find_spec("triton")   # None if truly not installed
_triton_patched = False
if _triton_spec is not None:
    # triton is installed — import normally and patch if DummyAutotuner missing
    try:
        import triton
        import triton.runtime
        if not hasattr(triton.runtime, "Autotuner"):
            class DummyAutotuner:
                def __init__(self, *a, **k): pass
                def tune(self, *a, **k): return None
            triton.runtime.Autotuner = DummyAutotuner
        _triton_patched = True
    except Exception:
        pass  # will fall through to teacher-free mode

# ─── Optional heavy imports ───────────────────────────────────────────────────
# StableVideoDiffusionPipeline is imported lazily inside __init__
# to avoid triton.__spec__ conflicts that happen when transformers
# calls importlib.util.find_spec("triton") at module load time.
TEACHER_AVAILABLE = False
try:
    import importlib.util as _dcheck
    if _dcheck.find_spec("diffusers") is not None:
        TEACHER_AVAILABLE = True
    else:
        print("diffusers not installed — teacher-free mode")
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  |  Teacher: {TEACHER_AVAILABLE}")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  WaveLatentBuffer
#     Maintains one oscillator per latent dimension per spatial position.
#     Fully vectorised — no Python loops in forward or update.
# ═════════════════════════════════════════════════════════════════════════════
class WaveLatentBuffer:
    """
    Vectorised WaveNeuron array over the latent space [C, H, W].

    Each voxel (c, h, w) has its own oscillator:
        z(t) = A * sin(2π·f·t + φ)

    Parameters are updated via a single gradient step on the current latent.
    Query at any real-valued t for temporal interpolation.

    Attributes
    ----------
    C, H, W  : latent channels, height, width
    A        : (C, H, W) amplitude tensor
    f        : (C, H, W) frequency tensor  [Hz]
    phi      : (C, H, W) phase tensor      [rad]
    lr       : update learning rate
    """

    def __init__(self,
                 C: int = 4, H: int = 64, W: int = 64,
                 init_freq_range=(0.5, 4.0),
                 lr: float = 0.15,
                 device: str = "cpu"):
        self.C, self.H, self.W = C, H, W
        self.lr   = lr
        self.dev  = device
        self._t   = 0.0          # current time (seconds)
        self._eml_history: list[float] = []   # for strip-chart display

        # Oscillator parameters — NOT nn.Parameters; updated manually
        rng = np.random.RandomState(42)
        lo, hi = init_freq_range
        self.A   = torch.zeros(C, H, W, device=device)
        self.f   = torch.tensor(
            rng.uniform(lo, hi, (C, H, W)), dtype=torch.float32, device=device)
        self.phi = torch.tensor(
            rng.uniform(0, 2*np.pi, (C, H, W)), dtype=torch.float32, device=device)

    # ── Query: latent at arbitrary time ──────────────────────────────────
    def predict(self, t: float) -> torch.Tensor:
        """Return the oscillator output at time t. Shape: (C, H, W)."""
        return self.A * torch.sin(2 * np.pi * self.f * t + self.phi)

    # ── Update: fit oscillators to a new observed latent ─────────────────
    @torch.no_grad()
    def update(self, latent: torch.Tensor, t: float) -> float:
        """
        One-step parameter update towards latent at time t.
        latent: (1, C, H, W) or (C, H, W)
        Returns the prediction error (scalar).
        """
        if latent.dim() == 4:
            latent = latent.squeeze(0)   # (C, H, W)

        predicted = self.predict(t)
        error     = latent - predicted                           # (C, H, W)

        # Gradient of A*sin(2πft+φ) w.r.t. each parameter:
        sin_val = torch.sin(2 * np.pi * self.f * t + self.phi)
        cos_val = torch.cos(2 * np.pi * self.f * t + self.phi)

        # Parameter updates (gradient ascent on negative MSE = descent on MSE)
        self.A   += self.lr * error * sin_val
        self.phi += self.lr * error * self.A * cos_val
        self.f   += self.lr * error * self.A * (2 * np.pi * t) * cos_val

        # Clamp to prevent explosion
        self.A   = self.A.clamp(-5.0, 5.0)
        self.f   = self.f.clamp(0.05, 20.0)

        self._t = t
        err = float(error.abs().mean())
        return err

    # ── EML signal: aggregate "brain wave" ───────────────────────────────
    def eml_scalar(self, t: float) -> float:
        """
        EML = |Z| − arg(Z)  but using magnitude only (no phase-wrap)
        Aggregate over all oscillators → one scalar per frame.
        """
        pred = self.predict(t)                    # (C, H, W)
        # Treat each (C, H, W) triplet as a complex Z
        # Use mean magnitude across all oscillators as the scalar signal
        mag = float(pred.abs().mean())
        self._eml_history.append(mag)
        if len(self._eml_history) > 200:
            self._eml_history.pop(0)
        return mag

    # ── Phase map for encoder conditioning ───────────────────────────────
    def phase_map(self, t: float) -> torch.Tensor:
        """
        Returns a (1, 1, H, W) tensor: the mean oscillation across channels,
        normalised to [-1, 1]. Used as an extra encoder input channel.
        """
        pred = self.predict(t)                    # (C, H, W)
        mean = pred.mean(dim=0, keepdim=True)     # (1, H, W)
        # Normalise by amplitude envelope
        scale = self.A.abs().mean(dim=0, keepdim=True).clamp(min=1e-4)
        return (mean / scale).unsqueeze(0).clamp(-1, 1)   # (1, 1, H, W)

    # ── Temporal smoothness: predicted vs. actual ────────────────────────
    def smoothness_loss(self, latent: torch.Tensor, t: float) -> torch.Tensor:
        """
        MSE between the buffer's prediction at t and the actual latent.
        Returned as a differentiable tensor for the training loss.
        latent: (1, C, H, W)
        """
        predicted = self.predict(t).unsqueeze(0)          # (1, C, H, W)
        return nn.functional.mse_loss(latent, predicted.detach())


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Student networks  (v2: encoder takes 4-channel input)
# ═════════════════════════════════════════════════════════════════════════════
class AdaptiveEncoderConv_v2(nn.Module):
    """
    Like AdaptiveEncoderConv but takes 4 input channels:
      channels 0-2: RGB (normalised to [0,1])
      channel    3: EML phase map from WaveLatentBuffer (normalised to [-1,1])

    The extra channel gives the encoder temporal context —
    it knows where it is in the oscillation cycle.
    """
    def __init__(self):
        super().__init__()
        # 4-channel input (was 3)
        self.conv1 = nn.Conv2d(4,   64,  kernel_size=4, stride=2, padding=1)  # 512→256
        self.conv2 = nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1)  # 256→128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 128→64
        self.conv4 = nn.Conv2d(256, 4,   kernel_size=3, stride=1, padding=1)  # 64→64, 4ch
        self.relu  = nn.ReLU()

    def forward(self, x):
        # x: (B, 4, 512, 512)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.conv4(x)   # (B, 4, 64, 64)


class AdaptiveDecoderConv(nn.Module):
    """Unchanged from unitvae4.py."""
    def __init__(self):
        super().__init__()
        self.conv_trans1 = nn.ConvTranspose2d(4,   256, kernel_size=3, stride=1, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1)
        self.conv_trans4 = nn.ConvTranspose2d(64,  3,   kernel_size=4, stride=2, padding=1)
        self.relu        = nn.ReLU()

    def forward(self, latent):
        x = self.relu(self.conv_trans1(latent))
        x = self.relu(self.conv_trans2(x))
        x = self.relu(self.conv_trans3(x))
        return torch.sigmoid(self.conv_trans4(x))


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Trainer  (with temporal smoothness loss)
# ═════════════════════════════════════════════════════════════════════════════
class AdaptiveVAETrainer_v2:
    """
    Training step with three loss components:
      latent_loss     : student latent ↔ teacher latent  (knowledge distillation)
      image_loss      : student reconstruction ↔ teacher reconstruction
      smoothness_loss : student latent ↔ wave-buffer prediction  (temporal)

    If teacher is None (teacher-free mode), uses only smoothness + image self-reconstruction.
    """
    def __init__(self, encoder, decoder, wave_buffer, teacher_vae=None,
                 lambda_smooth: float = 0.1):
        self.encoder     = encoder
        self.decoder     = decoder
        self.wave        = wave_buffer
        self.teacher     = teacher_vae
        self.lam_smooth  = lambda_smooth
        self.optimizer   = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4
        )
        self.loss_fn = nn.MSELoss()
        self.scaler  = torch.cuda.amp.GradScaler() if device == "cuda" else None

    def train_on_frame(self, image_tensor: torch.Tensor, t: float) -> dict:
        """
        image_tensor: (1, 3, 512, 512) in [0, 1]
        t           : current timestamp (float, seconds since start)
        Returns dict of loss components.
        """
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        # ── Phase map from wave buffer ─────────────────────────────────
        # (1, 1, 64, 64) → upsample to (1, 1, 512, 512) for encoder input
        phase_64   = self.wave.phase_map(t)                           # (1,1,64,64)
        phase_512  = nn.functional.interpolate(
            phase_64, size=(512, 512), mode='bilinear', align_corners=False
        )                                                              # (1,1,512,512)
        # Concatenate RGB + phase channel
        enc_input  = torch.cat([image_tensor, phase_512], dim=1)      # (1,4,512,512)

        # ── Teacher step (if available) ────────────────────────────────
        if self.teacher is not None:
            with torch.no_grad():
                teacher_latent = self.teacher.encode(
                    image_tensor.half()).latent_dist.sample().float()
                decoded = self.teacher.decode(teacher_latent.half(), num_frames=1).sample
                teacher_decoded = ((decoded / 2 + 0.5).clamp(0, 1)).float()
        else:
            teacher_latent  = None
            teacher_decoded = image_tensor   # self-supervision fallback

        # ── Student forward ────────────────────────────────────────────
        use_amp = (device == "cuda")
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_latent = self.encoder(enc_input)                      # (1,4,64,64)

            latent_loss = (
                self.loss_fn(pred_latent, teacher_latent)
                if teacher_latent is not None else torch.tensor(0.0, device=device)
            )

            pred_image  = self.decoder(pred_latent)                    # (1,3,512,512)
            image_loss  = self.loss_fn(pred_image, teacher_decoded)

            smooth_loss = self.wave.smoothness_loss(pred_latent, t)

            loss = latent_loss + image_loss + self.lam_smooth * smooth_loss

        # ── Backward ──────────────────────────────────────────────────
        if self.scaler:
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0
            )
            self.optimizer.step()

        # ── Update wave buffer with new latent ────────────────────────
        buf_err = self.wave.update(pred_latent.detach(), t)

        return dict(
            total   = float(loss),
            latent  = float(latent_loss),
            image   = float(image_loss),
            smooth  = float(smooth_loss),
            buf_err = buf_err,
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4.  GUI Application
# ═════════════════════════════════════════════════════════════════════════════
class LatentVideoFilter_v2:
    """
    Live webcam with:
      - Teach mode (distillation from SVD teacher or self-supervised)
      - Real-time latent wave strip chart
      - Temporal interpolation slider (query buffer between real frames)
      - Save / load model
    """

    def __init__(self, master):
        self.master = master
        self.master.title("UnitVAE 5 — Wave Latent Buffer  (PerceptionLab)")

        # ── Load teacher ──────────────────────────────────────────────
        self.teacher_vae = None
        if TEACHER_AVAILABLE:
            print("Loading Stable Video Diffusion…")
            try:
                # Import here (not at module level) so triton is fully settled
                from diffusers import StableVideoDiffusionPipeline
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=torch.float16
                ).to(device)
                self.teacher_vae = pipe.vae
                print("Teacher loaded.")
            except Exception as e:
                print(f"Teacher load failed ({e}), continuing without teacher.")

        # ── Student components ────────────────────────────────────────
        self.encoder     = AdaptiveEncoderConv_v2().to(device)
        self.decoder     = AdaptiveDecoderConv().to(device)
        self.wave_buffer = WaveLatentBuffer(C=4, H=64, W=64, device=device)
        self.trainer     = AdaptiveVAETrainer_v2(
            self.encoder, self.decoder, self.wave_buffer,
            teacher_vae=self.teacher_vae, lambda_smooth=0.1
        )

        # ── State ─────────────────────────────────────────────────────
        self.teach_mode   = False
        self.latest_frame = None
        self.frame_lock   = threading.Lock()
        self.start_time   = time.time()
        self.loss_log: list[dict] = []

        # ── GUI ───────────────────────────────────────────────────────
        self._build_gui()
        self._start_threads()

    # ── GUI layout ────────────────────────────────────────────────────
    def _build_gui(self):
        BTN = dict(bg="#333", fg="#eee", font=("Consolas", 9), relief="flat",
                   activebackground="#555", activeforeground="#fff")

        top = tk.Frame(self.master, bg="#111"); top.pack(side="top", fill="x")

        self.btn_teach = tk.Button(top, text="▶ Teach", command=self._toggle_teach, **BTN)
        self.btn_teach.pack(side="left", padx=4, pady=4)

        tk.Button(top, text="💾 Save", command=self._save, **BTN).pack(side="left", padx=4)
        tk.Button(top, text="📂 Load", command=self._load, **BTN).pack(side="left", padx=4)

        self.lbl_status = tk.Label(top, text="Ready", bg="#111", fg="#0f0",
                                   font=("Consolas", 9), anchor="w")
        self.lbl_status.pack(side="left", padx=8, fill="x", expand=True)

        # ── Main display area (video left, wave chart right) ──────────
        mid = tk.Frame(self.master, bg="#111"); mid.pack(fill="both", expand=True)

        self.lbl_video = tk.Label(mid, bg="#000"); self.lbl_video.pack(side="left")

        # Latent wave strip chart (200 × 256 canvas)
        wave_frame = tk.Frame(mid, bg="#111"); wave_frame.pack(side="left", fill="y", padx=6)
        tk.Label(wave_frame, text="Latent Wave (EML)",
                 bg="#111", fg="#8acc9a", font=("Consolas", 8)).pack()
        self.wave_canvas = tk.Canvas(wave_frame, width=200, height=256,
                                     bg="#0c0c10", highlightthickness=0)
        self.wave_canvas.pack()

        # Loss strip chart
        tk.Label(wave_frame, text="Training Loss",
                 bg="#111", fg="#cc8844", font=("Consolas", 8)).pack(pady=(8,0))
        self.loss_canvas = tk.Canvas(wave_frame, width=200, height=100,
                                     bg="#0c0c10", highlightthickness=0)
        self.loss_canvas.pack()

        # Temporal interpolation slider
        interp_frame = tk.Frame(self.master, bg="#111"); interp_frame.pack(fill="x")
        tk.Label(interp_frame, text="Temporal interpolation (β query offset)",
                 bg="#111", fg="#888", font=("Consolas", 8)).pack(side="left", padx=4)
        self.slider_interp = tk.Scale(
            interp_frame, from_=-1.0, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL, length=300, bg="#222", fg="#ccc",
            command=self._on_interp_slide
        )
        self.slider_interp.set(0.0)
        self.slider_interp.pack(side="left", padx=4)
        self.interp_offset = 0.0

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.master, textvariable=self.status_var,
                 relief="sunken", anchor="w", font=("Consolas", 8),
                 bg="#111", fg="#666").pack(side="bottom", fill="x")

    def _on_interp_slide(self, val):
        self.interp_offset = float(val)

    # ── Threads ───────────────────────────────────────────────────────
    def _start_threads(self):
        self.cap = cv2.VideoCapture(0)
        t = threading.Thread(target=self._training_loop, daemon=True)
        t.start()
        self._update_video()

    # ── Training loop (background thread) ────────────────────────────
    def _training_loop(self):
        while True:
            if self.teach_mode and self.latest_frame is not None:
                with self.frame_lock:
                    frame = self.latest_frame.copy()
                try:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    tensor = T.Compose([T.Resize((512, 512)), T.ToTensor()])(img)
                    tensor = tensor.unsqueeze(0).to(device)
                    t_now = time.time() - self.start_time
                    losses = self.trainer.train_on_frame(tensor, t_now)
                    self.loss_log.append(losses)
                    if len(self.loss_log) > 200:
                        self.loss_log.pop(0)
                    self.status_var.set(
                        f"teach  total={losses['total']:.4f}  "
                        f"img={losses['image']:.4f}  "
                        f"smooth={losses['smooth']:.4f}  "
                        f"buf_err={losses['buf_err']:.4f}"
                    )
                except Exception as e:
                    self.status_var.set(f"Training error: {e}")
            time.sleep(0.1)

    # ── Video update (main thread) ────────────────────────────────────
    def _update_video(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()

                try:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    tensor = T.Compose([T.Resize((512, 512)), T.ToTensor()])(img)
                    tensor = tensor.unsqueeze(0).to(device)

                    t_now   = time.time() - self.start_time
                    t_query = t_now + self.interp_offset

                    with torch.no_grad():
                        # Phase map at query time
                        phase_64  = self.wave_buffer.phase_map(t_query)
                        phase_512 = nn.functional.interpolate(
                            phase_64, size=(512, 512), mode='bilinear', align_corners=False
                        )
                        enc_input = torch.cat([tensor, phase_512], dim=1)
                        latent    = self.encoder(enc_input)

                        # If interpolation offset != 0: blend actual latent with
                        # wave-buffer prediction at the offset time
                        if abs(self.interp_offset) > 0.01:
                            alpha     = min(abs(self.interp_offset), 1.0)
                            predicted = self.wave_buffer.predict(t_query).unsqueeze(0)
                            latent    = (1 - alpha) * latent + alpha * predicted

                        recon = self.decoder(latent)

                    recon_np = (recon.cpu().squeeze(0).permute(1,2,0).numpy() * 255
                                ).clip(0, 255).astype(np.uint8)
                    disp = Image.fromarray(recon_np).resize((320, 240))
                    photo = ImageTk.PhotoImage(disp)
                    self.lbl_video.config(image=photo)
                    self.lbl_video.image = photo

                    # Update wave chart
                    eml = self.wave_buffer.eml_scalar(t_now)
                    self._draw_wave_chart()
                    self._draw_loss_chart()

                except Exception as e:
                    self.status_var.set(f"Render error: {e}")

        self.master.after(30, self._update_video)

    # ── Strip chart renderers ─────────────────────────────────────────
    def _draw_wave_chart(self):
        c = self.wave_canvas
        c.delete("all")
        h  = history = self.wave_buffer._eml_history
        if len(h) < 2:
            return
        W, H = 200, 256
        mn, mx = min(h), max(h)
        rng = (mx - mn) or 1.0
        pts = []
        for i, v in enumerate(h[-W:]):
            x = int(i * W / min(len(h), W))
            y = int(H - (v - mn) / rng * (H - 4) - 2)
            pts.extend([x, y])
        if len(pts) >= 4:
            c.create_line(*pts, fill="#5a8a6a", width=1, smooth=True)
        # Y label
        c.create_text(4, 4, anchor="nw", text=f"{mx:.3f}", fill="#444", font=("Consolas", 7))
        c.create_text(4, H-12, anchor="nw", text=f"{mn:.3f}", fill="#444", font=("Consolas", 7))

    def _draw_loss_chart(self):
        c = self.loss_canvas
        c.delete("all")
        if len(self.loss_log) < 2:
            c.create_text(100, 50, text="no loss yet", fill="#444", font=("Consolas", 8))
            return
        W, H = 200, 100
        totals = [d['total'] for d in self.loss_log[-W:]]
        mn, mx = 0, max(totals) or 1.0
        pts = []
        for i, v in enumerate(totals):
            x = int(i * W / min(len(totals), W))
            y = int(H - (v - mn) / mx * (H - 4) - 2)
            pts.extend([x, y])
        if len(pts) >= 4:
            c.create_line(*pts, fill="#cc8844", width=1)
        c.create_text(4, 4, anchor="nw", text=f"loss {totals[-1]:.4f}",
                      fill="#666", font=("Consolas", 7))

    # ── Teach toggle ──────────────────────────────────────────────────
    def _toggle_teach(self):
        self.teach_mode = not self.teach_mode
        if self.teach_mode:
            self.btn_teach.config(text="■ Stop teach", bg="#550000")
            self.status_var.set("Teach mode active")
        else:
            self.btn_teach.config(text="▶ Teach", bg="#333")
            self.status_var.set("Teach mode paused")

    # ── Save / load ───────────────────────────────────────────────────
    def _save(self):
        fn = filedialog.asksaveasfilename(
            title="Save model", defaultextension=".pth",
            filetypes=[("PyTorch", "*.pth")])
        if fn:
            torch.save({
                "encoder":     self.encoder.state_dict(),
                "decoder":     self.decoder.state_dict(),
                "wave_A":      self.wave_buffer.A.cpu(),
                "wave_f":      self.wave_buffer.f.cpu(),
                "wave_phi":    self.wave_buffer.phi.cpu(),
            }, fn)
            self.status_var.set(f"Saved → {fn}")

    def _load(self):
        fn = filedialog.askopenfilename(
            title="Load model", filetypes=[("PyTorch", "*.pth")])
        if fn:
            ck = torch.load(fn, map_location=device)
            self.encoder.load_state_dict(ck["encoder"])
            self.decoder.load_state_dict(ck["decoder"])
            if "wave_A" in ck:
                self.wave_buffer.A   = ck["wave_A"].to(device)
                self.wave_buffer.f   = ck["wave_f"].to(device)
                self.wave_buffer.phi = ck["wave_phi"].to(device)
            self.status_var.set(f"Loaded ← {fn}")

    def run(self):
        self.master.protocol("WM_DELETE_WINDOW", self._close)
        self.master.mainloop()

    def _close(self):
        if self.cap:
            self.cap.release()
        self.master.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    root.configure(bg="#111")
    app = LatentVideoFilter_v2(root)
    app.run()


if __name__ == "__main__":
    main()