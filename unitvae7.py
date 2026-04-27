"""
unitvae6.py  —  Hierarchical Wave Cortex
=========================================
PerceptionLab / Antti Luode, 2026

Extends unitvae5.py with a two-level visual-cortex-like architecture.

The problem with unitvae5:
  One wave buffer, one spatial scale, one update rate.
  Works. But the brain doesn't work like that.

The biological motivation:
  V1 / Retina     → fast, high-resolution, edge detection
  IT / Cortex     → slow, low-resolution, object/concept tracking
  Feedback loops  → slow context suppresses noise in fast layer
  Surprise signal → fast spike wakes up slow layer (adrenaline)

The habituation effect you noticed (cover → bright flash → slow settle):
  That's EXACTLY this system. The fast buffer sees the change immediately.
  The slow buffer takes longer. When slow finally catches up, it pulls
  fast back to equilibrium. That IS gain control / habituation.

Architecture:

    webcam (512×512×3)
         │
         ▼
  ┌──────────────────────────────────────────────────┐
  │  HierarchicalWaveCortex                          │
  │                                                  │
  │  FAST buffer (64×64, lr=0.15, f=1–8 Hz)         │
  │  SLOW buffer (16×16, lr=0.02, f=0.1–1 Hz)       │
  │                                                  │
  │  Bidirectional coupling:                         │
  │    slow.phase_map → conditions fast encoder      │
  │    fast.eml_spike → boosts slow lr (adrenaline)  │
  └──────────────────────────────────────────────────┘
         │
         ▼
  AdaptiveEncoderConv_v3  (5-channel input)
    ch 0-2: RGB
    ch   3: fast phase map (temporal momentum, 64×64)
    ch   4: slow context map (global structure, 16→64 upsampled)
         │
         ▼
  latent [4, 64, 64]
         │         │
         │   Cortex update:
         │     fast.update(latent, t)
         │     slow.update(pool(latent), t, adrenaline)
         ▼
  AdaptiveDecoderConv → reconstruction

GUI (three strip charts):
  - Fast EML  (green, spiky, high freq)
  - Slow EML  (orange, smooth, low freq)
  - Training loss (red)

Watching the two waves interact IS watching a two-level mind.
"""

import os, sys, types, threading, time
import cv2, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as T
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# ─── Triton patch (same fix as unitvae5) ─────────────────────────────────────
os.environ["DIFFUSERS_NO_IP_ADAPTER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import importlib.util as _iutil
_triton_spec = _iutil.find_spec("triton")
if _triton_spec is not None:
    try:
        import triton, triton.runtime
        if not hasattr(triton.runtime, "Autotuner"):
            class DummyAutotuner:
                def __init__(self, *a, **k): pass
                def tune(self, *a, **k): return None
            triton.runtime.Autotuner = DummyAutotuner
    except Exception:
        pass

# ─── Optional teacher ────────────────────────────────────────────────────────
TEACHER_AVAILABLE = False
try:
    if _iutil.find_spec("diffusers") is not None:
        TEACHER_AVAILABLE = True
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  |  Teacher: {TEACHER_AVAILABLE}")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  WaveLatentBuffer  (same as unitvae5, unchanged)
# ═════════════════════════════════════════════════════════════════════════════
class WaveLatentBuffer:
    """
    Vectorised WaveNeuron array: one oscillator per (C, H, W) voxel.
        z(t) = A * sin(2π·f·t + φ)
    """
    def __init__(self, C=4, H=64, W=64,
                 freq_range=(0.5, 4.0), lr=0.15, device="cpu"):
        self.C, self.H, self.W = C, H, W
        self.lr  = lr
        self.dev = device
        self._eml_history: list[float] = []

        rng = np.random.RandomState(42)
        lo, hi = freq_range
        self.A   = torch.zeros(C, H, W, device=device)
        self.f   = torch.tensor(
            rng.uniform(lo, hi, (C, H, W)), dtype=torch.float32, device=device)
        self.phi = torch.tensor(
            rng.uniform(0, 2*np.pi, (C, H, W)), dtype=torch.float32, device=device)

    @torch.no_grad()
    def predict(self, t: float) -> torch.Tensor:
        return self.A * torch.sin(2 * np.pi * self.f * t + self.phi)

    @torch.no_grad()
    def update(self, latent: torch.Tensor, t: float,
               lr_scale: float = 1.0) -> float:
        """
        lr_scale > 1 = "adrenaline" from fast buffer spike.
        """
        if latent.dim() == 4:
            latent = latent.squeeze(0)
        predicted = self.predict(t)
        error     = latent - predicted
        lr        = self.lr * lr_scale

        sin_val = torch.sin(2 * np.pi * self.f * t + self.phi)
        cos_val = torch.cos(2 * np.pi * self.f * t + self.phi)

        self.A   = (self.A   + lr * error * sin_val).clamp(-5., 5.)
        self.phi = self.phi  + lr * error * self.A * cos_val
        self.f   = (self.f   + lr * error * self.A * (2*np.pi*t) * cos_val
                    ).clamp(0.05, 20.)

        err = float(error.abs().mean())
        return err

    @torch.no_grad()
    def phase_map(self, t: float) -> torch.Tensor:
        """(1, 1, H, W) normalised to [-1, 1]."""
        pred  = self.predict(t)
        mean  = pred.mean(dim=0, keepdim=True)
        scale = self.A.abs().mean(dim=0, keepdim=True).clamp(min=1e-4)
        return (mean / scale).unsqueeze(0).clamp(-1, 1)

    @torch.no_grad()
    def eml_scalar(self, t: float) -> float:
        pred = self.predict(t)
        mag  = float(pred.abs().mean())
        self._eml_history.append(mag)
        if len(self._eml_history) > 300:
            self._eml_history.pop(0)
        return mag

    @torch.no_grad()
    def smoothness_loss(self, latent: torch.Tensor, t: float) -> torch.Tensor:
        predicted = self.predict(t).unsqueeze(0)
        return nn.functional.mse_loss(latent, predicted.detach())


# ═════════════════════════════════════════════════════════════════════════════
# 2.  HierarchicalWaveCortex  ← THE NEW THING
# ═════════════════════════════════════════════════════════════════════════════
class HierarchicalWaveCortex:
    """
    Two-level wave buffer system modelled on the visual cortex hierarchy.

    Fast buffer (V1 / retinal analogue):
        64×64 spatial, high frequencies (1–8 Hz), fast update rate
        Tracks immediate frame-to-frame changes.
        EML signal is spiky and reactive.

    Slow buffer (IT / cortex analogue):
        16×16 spatial, low frequencies (0.1–1 Hz), slow update rate
        Tracks global scene context and long-term dynamics.
        EML signal is smooth and persistent.

    Bidirectional coupling:
        TOP-DOWN:  slow.phase_map → feeds into encoder as context channel
                   This tells the encoder "globally, the scene is stable"
                   and suppresses noise in the fast layer.

        BOTTOM-UP: fast.eml spike → temporarily boosts slow.lr (adrenaline)
                   When fast sees something surprising, it wakes up slow.
                   This is the iris/habituation effect you observed.

    The "letters not scrambled" observation:
        The slow buffer builds a stable, low-frequency spatial prior over
        seconds of viewing. Text has a specific low-frequency signature
        (regular horizontal bands). The slow context locks onto this and
        communicates it top-down, giving the fast layer a stable frame
        within which to resolve fine structure. Standard diffusion lacks
        this long-horizon contextual prior entirely.
    """

    # Adrenaline parameters
    SPIKE_THRESHOLD   = 0.3    # fast EML deviation that triggers adrenaline
    ADRENALINE_GAIN   = 5.0    # how much to boost slow lr on spike
    ADRENALINE_DECAY  = 0.85   # exponential decay of adrenaline per frame

    def __init__(self, device: str = "cpu"):
        self.dev = device

        # Fast buffer: 64×64, f=1–8 Hz (detail layer)
        self.fast = WaveLatentBuffer(
            C=4, H=64, W=64,
            freq_range=(1.0, 8.0), lr=0.15, device=device
        )

        # Slow buffer: 16×16, f=0.1–1 Hz (context layer)
        self.slow = WaveLatentBuffer(
            C=4, H=16, W=16,
            freq_range=(0.1, 1.0), lr=0.02, device=device
        )

        # Pooling: 64×64 → 16×16
        self._pool = nn.AdaptiveAvgPool2d((16, 16))

        # Adrenaline state
        self._adrenaline       = 0.0
        self._fast_eml_mean    = 0.5   # running mean for spike detection
        self._fast_eml_std     = 0.2   # running std

        # History for GUI
        self.fast_history: list[float] = []
        self.slow_history: list[float] = []

    # ── Update both buffers with new latent ──────────────────────────────
    @torch.no_grad()
    def update(self, latent: torch.Tensor, t: float) -> dict:
        """
        latent: (1, 4, 64, 64)
        Returns diagnostics dict.
        """
        # ── Fast buffer update ──────────────────────────────────────────
        fast_err = self.fast.update(latent, t)

        # ── Adrenaline computation ──────────────────────────────────────
        fast_eml = self.fast.eml_scalar(t)
        # Running mean/std update (exponential)
        self._fast_eml_mean = 0.95 * self._fast_eml_mean + 0.05 * fast_eml
        deviation = abs(fast_eml - self._fast_eml_mean)
        self._fast_eml_std  = 0.95 * self._fast_eml_std  + 0.05 * deviation

        # Spike = deviation > threshold * std
        normalised_spike = (deviation / (self._fast_eml_std + 1e-6))
        if normalised_spike > self.SPIKE_THRESHOLD:
            self._adrenaline = min(
                self._adrenaline + normalised_spike * 0.5,
                self.ADRENALINE_GAIN
            )

        # ── Slow buffer update (with adrenaline) ───────────────────────
        # Downsample latent to 16×16 for slow buffer
        latent_pooled = self._pool(latent)           # (1, 4, 16, 16)
        lr_scale = 1.0 + self._adrenaline
        slow_err  = self.slow.update(latent_pooled, t, lr_scale=lr_scale)

        # Adrenaline decays each frame
        self._adrenaline *= self.ADRENALINE_DECAY

        # ── Record slow EML ────────────────────────────────────────────
        slow_eml = self.slow.eml_scalar(t)
        self.fast_history.append(fast_eml)
        self.slow_history.append(slow_eml)
        if len(self.fast_history) > 300: self.fast_history.pop(0)
        if len(self.slow_history) > 300: self.slow_history.pop(0)

        return dict(
            fast_err   = fast_err,
            slow_err   = slow_err,
            fast_eml   = fast_eml,
            slow_eml   = slow_eml,
            adrenaline = self._adrenaline,
        )

    # ── Phase maps for encoder conditioning ──────────────────────────────
    @torch.no_grad()
    def fast_phase_512(self, t: float) -> torch.Tensor:
        """Fast phase map upsampled to 512×512. Shape: (1, 1, 512, 512)."""
        pm = self.fast.phase_map(t)            # (1, 1, 64, 64)
        return nn.functional.interpolate(
            pm, size=(512, 512), mode='bilinear', align_corners=False
        )

    @torch.no_grad()
    def slow_context_512(self, t: float) -> torch.Tensor:
        """Slow context map upsampled to 512×512. Shape: (1, 1, 512, 512)."""
        pm = self.slow.phase_map(t)            # (1, 1, 16, 16)
        return nn.functional.interpolate(
            pm, size=(512, 512), mode='bilinear', align_corners=False
        )

    # ── Smoothness loss (from fast buffer only) ───────────────────────────
    def smoothness_loss(self, latent: torch.Tensor, t: float) -> torch.Tensor:
        return self.fast.smoothness_loss(latent, t)

    # ── Temporal interpolation ─────────────────────────────────────────────
    @torch.no_grad()
    def predict_latent(self, t: float) -> torch.Tensor:
        """Blend fast and slow predictions at query time."""
        fast_pred = self.fast.predict(t).unsqueeze(0)    # (1,4,64,64)
        slow_pred = self.slow.predict(t)                 # (4,16,16)
        slow_up   = nn.functional.interpolate(
            slow_pred.unsqueeze(0), size=(64, 64),
            mode='bilinear', align_corners=False
        )
        # Blend: slow anchors the low-frequency structure
        return 0.7 * fast_pred + 0.3 * slow_up


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Networks
# ═════════════════════════════════════════════════════════════════════════════
class AdaptiveEncoderConv_v3(nn.Module):
    """
    5-channel input:
      ch 0-2: RGB
      ch   3: fast phase map  (temporal momentum signal)
      ch   4: slow context map (global structure signal)

    The encoder now sees both "where the oscillators are right now" (fast)
    and "what the scene has been like over the last few seconds" (slow).
    This is the direct analogue of cortical top-down feedback.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5,   64,  kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 4,   kernel_size=3, stride=1, padding=1)
        self.relu  = nn.ReLU()

    def forward(self, x):
        # x: (B, 5, 512, 512)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.conv4(x)   # (B, 4, 64, 64)


class AdaptiveDecoderConv(nn.Module):
    """Unchanged from unitvae4/5."""
    def __init__(self):
        super().__init__()
        self.conv_trans1 = nn.ConvTranspose2d(4,   256, 3, 1, 1)
        self.conv_trans2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv_trans3 = nn.ConvTranspose2d(128, 64,  4, 2, 1)
        self.conv_trans4 = nn.ConvTranspose2d(64,  3,   4, 2, 1)
        self.relu        = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.conv_trans1(z))
        x = self.relu(self.conv_trans2(x))
        x = self.relu(self.conv_trans3(x))
        return torch.sigmoid(self.conv_trans4(x))


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Trainer
# ═════════════════════════════════════════════════════════════════════════════
class CortexTrainer:
    def __init__(self, encoder, decoder, cortex,
                 teacher_vae=None, lambda_smooth=0.1):
        self.encoder  = encoder
        self.decoder  = decoder
        self.cortex   = cortex
        self.teacher  = teacher_vae
        self.lam      = lambda_smooth
        self.opt      = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4
        )
        self.loss_fn = nn.MSELoss()
        self.scaler  = torch.cuda.amp.GradScaler() if device == "cuda" else None

    def step(self, image_tensor: torch.Tensor, t: float) -> dict:
        """image_tensor: (1, 3, 512, 512) in [0, 1]"""
        self.encoder.train(); self.decoder.train()
        self.opt.zero_grad()

        # ── Conditioning channels from cortex ─────────────────────────
        fast_ch = self.cortex.fast_phase_512(t)       # (1,1,512,512)
        slow_ch = self.cortex.slow_context_512(t)     # (1,1,512,512)
        enc_in  = torch.cat([image_tensor, fast_ch, slow_ch], dim=1)  # (1,5,512,512)

        # ── Teacher (if available) ─────────────────────────────────────
        if self.teacher is not None:
            with torch.no_grad():
                tl = self.teacher.encode(
                    image_tensor.half()).latent_dist.sample().float()
                td_raw = self.teacher.decode(tl.half(), num_frames=1).sample
                td     = ((td_raw / 2 + 0.5).clamp(0, 1)).float()
        else:
            tl = None
            td = image_tensor

        # ── Forward ───────────────────────────────────────────────────
        use_amp = (device == "cuda")
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_latent = self.encoder(enc_in)

            lat_loss  = (self.loss_fn(pred_latent, tl)
                         if tl is not None else torch.tensor(0., device=device))
            pred_img  = self.decoder(pred_latent)
            img_loss  = self.loss_fn(pred_img, td)
            smo_loss  = self.cortex.smoothness_loss(pred_latent, t)
            loss      = lat_loss + img_loss + self.lam * smo_loss

        # ── Backward ──────────────────────────────────────────────────
        if self.scaler:
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0)
            self.scaler.step(self.opt); self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0)
            self.opt.step()

        # ── Update cortex with new latent ──────────────────────────────
        diag = self.cortex.update(pred_latent.detach(), t)

        return dict(
            total    = float(loss),
            image    = float(img_loss),
            smooth   = float(smo_loss),
            **diag,
        )


# ═════════════════════════════════════════════════════════════════════════════
# 5.  GUI
# ═════════════════════════════════════════════════════════════════════════════
class HierarchicalCortexApp:
    """
    Live webcam with hierarchical wave cortex.
    Three strip charts: Fast EML / Slow EML / Loss
    """

    def __init__(self, master):
        self.master     = master
        self.master.title("UnitVAE 6 — Hierarchical Wave Cortex  (PerceptionLab)")
        self.start_time = time.time()

        # ── Teacher ────────────────────────────────────────────────────
        self.teacher_vae = None
        if TEACHER_AVAILABLE:
            print("Loading Stable Video Diffusion…")
            try:
                from diffusers import StableVideoDiffusionPipeline
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=torch.float16
                ).to(device)
                self.teacher_vae = pipe.vae
                print("Teacher loaded.")
            except Exception as e:
                print(f"Teacher failed ({e}) — self-supervised mode.")

        # ── Student + cortex ───────────────────────────────────────────
        self.encoder = AdaptiveEncoderConv_v3().to(device)
        self.decoder = AdaptiveDecoderConv().to(device)
        self.cortex  = HierarchicalWaveCortex(device=device)
        self.trainer = CortexTrainer(
            self.encoder, self.decoder, self.cortex,
            teacher_vae=self.teacher_vae, lambda_smooth=0.1
        )

        # ── State ──────────────────────────────────────────────────────
        self.teach_mode   = False
        self.latest_frame = None
        self.frame_lock   = threading.Lock()
        self.loss_log:    list[dict] = []
        self.interp_off   = 0.0

        self._build_gui()
        self._start()

    # ── GUI ────────────────────────────────────────────────────────────
    def _build_gui(self):
        BTN = dict(bg="#222", fg="#eee", font=("Consolas", 9),
                   relief="flat", activebackground="#444")

        # ── Top bar ───────────────────────────────────────────────────
        top = tk.Frame(self.master, bg="#111"); top.pack(side="top", fill="x")
        self.btn_teach = tk.Button(top, text="▶ Teach",
                                   command=self._toggle_teach, **BTN)
        self.btn_teach.pack(side="left", padx=4, pady=4)
        tk.Button(top, text="💾 Save", command=self._save, **BTN).pack(side="left", padx=4)
        tk.Button(top, text="📂 Load", command=self._load, **BTN).pack(side="left", padx=4)
        self.lbl_info = tk.Label(top, text="Ready", bg="#111", fg="#0f0",
                                 font=("Consolas", 8), anchor="w")
        self.lbl_info.pack(side="left", fill="x", expand=True, padx=8)

        # ── Main area ─────────────────────────────────────────────────
        mid = tk.Frame(self.master, bg="#111"); mid.pack(fill="both", expand=True)

        self.lbl_video = tk.Label(mid, bg="#000")
        self.lbl_video.pack(side="left", padx=4, pady=4)

        # Right panel: three charts
        right = tk.Frame(mid, bg="#111"); right.pack(side="left", fill="both", expand=True)

        def _chart(parent, label, color):
            f = tk.Frame(parent, bg="#111"); f.pack(fill="x", pady=2)
            tk.Label(f, text=label, bg="#111", fg=color,
                     font=("Consolas", 8)).pack(anchor="w", padx=4)
            c = tk.Canvas(f, width=260, height=80, bg="#0c0c10",
                          highlightthickness=0); c.pack(padx=4)
            return c

        self.c_fast = _chart(right, "▲ FAST EML  (V1 / retina)",    "#5a8a6a")
        self.c_slow = _chart(right, "▼ SLOW EML  (IT / cortex)",    "#cc8844")
        self.c_loss = _chart(right, "   Training Loss",             "#cc5555")

        # Adrenaline meter
        adr_f = tk.Frame(right, bg="#111"); adr_f.pack(fill="x", padx=4, pady=2)
        tk.Label(adr_f, text="⚡ Adrenaline", bg="#111", fg="#cc4444",
                 font=("Consolas", 8)).pack(side="left")
        self.adr_bar = tk.Canvas(adr_f, width=200, height=14,
                                 bg="#0c0c10", highlightthickness=0)
        self.adr_bar.pack(side="left", padx=4)

        # Temporal interpolation slider
        bot = tk.Frame(self.master, bg="#111"); bot.pack(fill="x")
        tk.Label(bot, text="β offset", bg="#111", fg="#666",
                 font=("Consolas", 8)).pack(side="left", padx=4)
        sl = tk.Scale(bot, from_=-1.0, to=1.0, resolution=0.01,
                      orient=tk.HORIZONTAL, length=260, bg="#1a1a1a", fg="#888",
                      command=lambda v: setattr(self, "interp_off", float(v)))
        sl.set(0.0); sl.pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.master, textvariable=self.status_var, relief="sunken",
                 anchor="w", font=("Consolas", 7), bg="#0a0a0a", fg="#555"
                 ).pack(side="bottom", fill="x")

    # ── Threads ────────────────────────────────────────────────────────
    def _start(self):
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self._train_loop, daemon=True).start()
        self._update_video()

    def _train_loop(self):
        while True:
            if self.teach_mode and self.latest_frame is not None:
                with self.frame_lock:
                    frame = self.latest_frame.copy()
                try:
                    img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    tensor = T.Compose([T.Resize((512,512)), T.ToTensor()])(img)
                    tensor = tensor.unsqueeze(0).to(device)
                    t_now  = time.time() - self.start_time
                    d      = self.trainer.step(tensor, t_now)
                    self.loss_log.append(d)
                    if len(self.loss_log) > 300: self.loss_log.pop(0)
                    self.status_var.set(
                        f"total={d['total']:.4f}  img={d['image']:.4f}  "
                        f"smooth={d['smooth']:.4f}  "
                        f"fast_eml={d['fast_eml']:.3f}  "
                        f"slow_eml={d['slow_eml']:.3f}  "
                        f"adrenaline={d['adrenaline']:.2f}"
                    )
                except Exception as e:
                    self.status_var.set(f"Train error: {e}")
            time.sleep(0.1)

    # ── Video + rendering ──────────────────────────────────────────────
    def _update_video(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                try:
                    img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    tensor = T.Compose([T.Resize((512,512)), T.ToTensor()])(img)
                    tensor = tensor.unsqueeze(0).to(device)
                    t_now  = time.time() - self.start_time
                    t_q    = t_now + self.interp_off

                    with torch.no_grad():
                        fast_ch = self.cortex.fast_phase_512(t_q)
                        slow_ch = self.cortex.slow_context_512(t_q)
                        enc_in  = torch.cat([tensor, fast_ch, slow_ch], dim=1)
                        latent  = self.encoder(enc_in)

                        if abs(self.interp_off) > 0.01:
                            alpha   = min(abs(self.interp_off), 1.0)
                            pred    = self.cortex.predict_latent(t_q)
                            latent  = (1-alpha)*latent + alpha*pred

                        recon = self.decoder(latent)

                    recon_np = (recon.cpu().squeeze(0).permute(1,2,0).numpy()*255
                                ).clip(0,255).astype(np.uint8)
                    photo = ImageTk.PhotoImage(
                        Image.fromarray(recon_np).resize((320, 240)))
                    self.lbl_video.config(image=photo)
                    self.lbl_video.image = photo

                    self._draw_charts()

                except Exception as e:
                    self.status_var.set(f"Render: {e}")

        self.master.after(30, self._update_video)

    # ── Chart rendering ────────────────────────────────────────────────
    def _sparkline(self, canvas, data: list, color: str, W=260, H=80):
        canvas.delete("all")
        if len(data) < 2: return
        mn, mx = min(data), max(data)
        rng    = (mx - mn) or 1e-6
        pts    = []
        n      = min(len(data), W)
        seg    = data[-n:]
        for i, v in enumerate(seg):
            x = int(i * W / n)
            y = int(H - (v-mn)/rng*(H-4) - 2)
            pts.extend([x, y])
        if len(pts) >= 4:
            canvas.create_line(*pts, fill=color, width=1, smooth=True)
        canvas.create_text(4, 4, anchor="nw",
                           text=f"{mx:.3f}", fill="#333", font=("Consolas", 7))
        canvas.create_text(4, H-12, anchor="nw",
                           text=f"{mn:.3f}", fill="#333", font=("Consolas", 7))

    def _draw_charts(self):
        self._sparkline(self.c_fast, self.cortex.fast_history, "#5a8a6a")
        self._sparkline(self.c_slow, self.cortex.slow_history, "#cc8844")

        # Loss chart
        if self.loss_log:
            totals = [d["total"] for d in self.loss_log]
            self._sparkline(self.c_loss, totals, "#cc5555")

        # Adrenaline bar
        adr  = min(self.cortex._adrenaline / self.cortex.ADRENALINE_GAIN, 1.0)
        w    = int(adr * 196)
        self.adr_bar.delete("all")
        if w > 0:
            # Colour: green → yellow → red
            r = int(min(w * 255 / 100, 255))
            g = int(max(255 - w * 2, 0))
            col = f"#{r:02x}{g:02x}00"
            self.adr_bar.create_rectangle(0, 0, w, 14, fill=col, outline="")

    # ── Toggle ─────────────────────────────────────────────────────────
    def _toggle_teach(self):
        self.teach_mode = not self.teach_mode
        if self.teach_mode:
            self.btn_teach.config(text="■ Stop", bg="#550000")
            self.lbl_info.config(text="Teaching…")
        else:
            self.btn_teach.config(text="▶ Teach", bg="#222")
            self.lbl_info.config(text="Paused")

    # ── Save / load ────────────────────────────────────────────────────
    def _save(self):
        fn = filedialog.asksaveasfilename(
            title="Save cortex model", defaultextension=".pth",
            filetypes=[("PyTorch", "*.pth")])
        if fn:
            torch.save({
                "encoder":  self.encoder.state_dict(),
                "decoder":  self.decoder.state_dict(),
                "fast_A":   self.cortex.fast.A.cpu(),
                "fast_f":   self.cortex.fast.f.cpu(),
                "fast_phi": self.cortex.fast.phi.cpu(),
                "slow_A":   self.cortex.slow.A.cpu(),
                "slow_f":   self.cortex.slow.f.cpu(),
                "slow_phi": self.cortex.slow.phi.cpu(),
            }, fn)
            self.status_var.set(f"Saved → {fn}")

    def _load(self):
        fn = filedialog.askopenfilename(
            title="Load cortex model", filetypes=[("PyTorch", "*.pth")])
        if fn:
            ck = torch.load(fn, map_location=device)
            self.encoder.load_state_dict(ck["encoder"])
            self.decoder.load_state_dict(ck["decoder"])
            for buf, key in [(self.cortex.fast, "fast"), (self.cortex.slow, "slow")]:
                if f"{key}_A" in ck:
                    buf.A   = ck[f"{key}_A"].to(device)
                    buf.f   = ck[f"{key}_f"].to(device)
                    buf.phi = ck[f"{key}_phi"].to(device)
            self.status_var.set(f"Loaded ← {fn}")

    def run(self):
        self.master.protocol("WM_DELETE_WINDOW", self._close)
        self.master.mainloop()

    def _close(self):
        if self.cap: self.cap.release()
        self.master.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    root.configure(bg="#111")
    HierarchicalCortexApp(root).run()

if __name__ == "__main__":
    main()