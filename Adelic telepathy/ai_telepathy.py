"""
ai_telepathy.py — AI-to-AI Knowledge Transfer via Prime Waves
==============================================================
PerceptionLab / Antti Luode, 2026

Pipeline
--------
  1.  Create two synthetic images  (AI-1's "knowledge")
  2.  Train AI-1  (Janus: stores both images phase-orthogonally)
  3.  AI-1 "daydreams": samples its output at anchor points
  4.  WaveEncoder packs samples into a 1-D prime-frequency wave
      S(t) = Σ_k  v_k · cos( 2π · BASE_FREQ · log(p_k) · t )
  5.  Noise added to simulate transmission channel
  6.  WaveDecoder recovers samples via least-squares (prime DFT)
  7.  AI-2 trains ONLY on the decoded samples  (no weight copying!)
  8.  Visualise: originals / AI-1 output / wave / AI-2 output

Honest assessment
-----------------
  - The prime-frequency codec works with >99% fidelity for N≤150 values
  - AI-2 receives a compressed (32 anchor points × 2 phases) signal
  - AI-2 generalises from that sparse data to fill in the full image
  - Quality depends on AI-2's ability to interpolate from anchor points
  - This is not magic; it is compressed sensing + neural interpolation
  - The "prime frequencies" provide a well-conditioned basis for encoding;
    equal-spacing DFT would work too, but prime spacing avoids any
    accidental harmonic resonances between carriers
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, os, sys

# Add local directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))
from wave_codec  import WaveCodec
from janus_numpy import JanusNumpy


# ─────────────────────────────────────────────────────────────────
# 1.  Synthetic images
# ─────────────────────────────────────────────────────────────────
def make_image_a(size: int = 32) -> np.ndarray:
    """Image A: concentric coloured rings (warm)."""
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    r    = np.sqrt(x**2 + y**2)
    red  = 0.5 + 0.5 * np.cos(r * 8)
    grn  = 0.5 + 0.5 * np.cos(r * 8 + 2*np.pi/3)
    blu  = 0.3 * (1 - r)
    img  = np.stack([red, grn, blu], axis=-1).clip(0, 1)
    return img.astype(np.float32)


def make_image_b(size: int = 32) -> np.ndarray:
    """Image B: diagonal wave grid (cool)."""
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    wave = 0.5 + 0.5 * np.sin(x * 6 + y * 6)
    red  = 0.1 * wave
    grn  = 0.4 * wave
    blu  = 0.8 + 0.2 * np.cos(x * 4 - y * 4)
    img  = np.stack([red, grn, blu], axis=-1).clip(0, 1)
    return img.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# 2.  Evaluation helpers
# ─────────────────────────────────────────────────────────────────
def full_render(net: JanusNumpy, size: int = 32, phase: float = 0.0) -> np.ndarray:
    """Render the full image from a network at a given phase."""
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    coords = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)
    rgb = net.predict(coords, phase=phase)
    return rgb.reshape(size, size, 3).clip(0, 1)


def mse_image(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target)**2))


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = mse_image(pred, target)
    return 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-12))


# ─────────────────────────────────────────────────────────────────
# 3.  Visualisation
# ─────────────────────────────────────────────────────────────────
def make_plots(img_a, img_b, ai1, ai2,
               wave, wave_noisy, codec, fidelity,
               loss_ai1, loss_ai2,
               out_path: str = "telepathy_result.png") -> None:

    fig = plt.figure(figsize=(18, 13), facecolor='#0c0c10')
    gs  = gridspec.GridSpec(4, 5, figure=fig,
          left=0.04, right=0.98, top=0.93, bottom=0.05,
          hspace=0.55, wspace=0.30)
    fig.suptitle('AI Telepathy — Wave-Based Knowledge Transfer  (PerceptionLab)',
                 color='#ccc', fontsize=13, fontfamily='monospace')

    def ax(r, c, **kw):
        a = fig.add_subplot(gs[r, c], **kw)
        a.set_facecolor('#0c0c10')
        for sp in a.spines.values(): sp.set_color('#2a2a35')
        a.tick_params(colors='#555', labelsize=7)
        return a

    def show(a, img, title, psnr_val=None):
        a.imshow(img, interpolation='nearest')
        lbl = title
        if psnr_val is not None:
            lbl += f'\nPSNR {psnr_val:.1f} dB'
        a.set_title(lbl, color='#bbb', fontsize=8, pad=3)
        a.axis('off')

    # Row 0: original images
    show(ax(0,0), img_a, 'Original A\n(ring pattern)')
    show(ax(0,1), img_b, 'Original B\n(wave pattern)')

    # Row 0 col 2: wave signal
    t_disp = codec.t
    ax0 = ax(0, 2)
    ax0.plot(t_disp[:2048], wave[:2048], color='#5a8a6a', lw=0.6, label='clean')
    ax0.plot(t_disp[:2048], wave_noisy[:2048], color='#cc8844', lw=0.5, alpha=0.7, label='noisy')
    ax0.set_title(f'Prime-frequency wave\n({codec.N} carriers, fid={fidelity:.1f}%)',
                  color='#bbb', fontsize=8, pad=3)
    ax0.set_xlabel('t (s)', color='#555', fontsize=7)
    ax0.legend(fontsize=6, facecolor='#111', edgecolor='#333', labelcolor='#aaa')

    # Frequency spectrum of wave
    ax_spec = ax(0, 3)
    freqs_hz = codec.omegas / (2*np.pi)
    # Mark carrier frequencies on spectrum
    fft_wave = np.abs(np.fft.rfft(wave_noisy))
    fft_f    = np.fft.rfftfreq(len(codec.t), 1.0/codec.sr)
    ax_spec.semilogy(fft_f[:len(fft_f)//4], fft_wave[:len(fft_f)//4]+1e-10,
                     color='#5a8a6a', lw=0.7)
    # Mark prime carriers
    for w_hz in freqs_hz:
        ax_spec.axvline(w_hz, color='#cc5555', lw=0.4, alpha=0.5)
    ax_spec.set_title('Wave spectrum\n(red = prime carriers)',
                       color='#bbb', fontsize=8, pad=3)
    ax_spec.set_xlabel('Hz', color='#555', fontsize=7)
    ax_spec.set_xlim(0, freqs_hz.max()*1.1)

    # Training loss curves
    ax_loss = ax(0, 4)
    ax_loss.semilogy(loss_ai1, color='#5a8a6a', lw=1.0, label='AI-1 (direct)', alpha=0.8)
    ax_loss.semilogy(loss_ai2, color='#cc8844', lw=1.0, label='AI-2 (telepathy)', alpha=0.8)
    ax_loss.set_title('Training loss\n(AI-1 vs AI-2)', color='#bbb', fontsize=8, pad=3)
    ax_loss.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='#aaa')
    ax_loss.set_xlabel('iteration', color='#555', fontsize=7)

    # Rows 1-2: AI-1 vs AI-2 outputs at different phases
    phases     = [0.0, np.pi/4, np.pi/2]
    phase_lbls = ['φ=0  (A)', 'φ=π/4 (blend)', 'φ=π/2 (B)']
    size       = img_a.shape[0]

    for col, (phi, lbl) in enumerate(zip(phases, phase_lbls)):
        r1 = full_render(ai1, size, phi)
        r2 = full_render(ai2, size, phi)
        
        target   = img_a if phi == 0.0 else (img_b if phi == np.pi/2 else None)
        psnr1    = psnr(r1, target) if target is not None else None
        psnr2    = psnr(r2, target) if target is not None else None

        show(ax(1, col+1), r1,
             f'AI-1 output\n{lbl}', psnr1)
        show(ax(2, col+1), r2,
             f'AI-2 output\n{lbl}', psnr2)

    # AI-1 label on row 1
    ax(1,0).text(0.5, 0.5, 'AI-1\n(direct\ntraining)',
                 ha='center', va='center', color='#5a8a6a',
                 fontsize=11, transform=fig.add_subplot(gs[1,0]).transAxes)
    fig.add_subplot(gs[1,0]).axis('off')
    ax(2,0).text(0.5, 0.5, 'AI-2\n(wave\ntelepathy)',
                 ha='center', va='center', color='#cc8844',
                 fontsize=11, transform=fig.add_subplot(gs[2,0]).transAxes)
    fig.add_subplot(gs[2,0]).axis('off')

    # Row 3: difference maps
    for col, (phi, lbl) in enumerate(zip([0.0, np.pi/2], ['A', 'B'])):
        r1  = full_render(ai1, size, phi)
        r2  = full_render(ai2, size, phi)
        diff = np.abs(r1 - r2)
        ax_d = ax(3, col+1)
        ax_d.imshow(diff * 4, interpolation='nearest', cmap='hot')
        ax_d.set_title(f'|AI-1 − AI-2| (×4)\nImage {lbl}\n'
                       f'RMSE={np.sqrt(np.mean(diff**2)):.4f}',
                       color='#bbb', fontsize=8, pad=3)
        ax_d.axis('off')

    # Summary text
    summary = (f"Codec: {codec.N} prime carriers  |  "
               f"Transmission fidelity: {fidelity:.2f}%  |  "
               f"Channel: T={codec.duration:.0f}s @ {codec.sr} Hz  |  "
               f"AI-2 train data: {codec.N} decoded samples (no raw weights)")
    fig.text(0.5, 0.012, summary, ha='center', color='#666', fontsize=8,
             fontfamily='monospace')

    fig.savefig(out_path, dpi=150, facecolor='#0c0c10')
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────
# 4.  Main pipeline
# ─────────────────────────────────────────────────────────────────
def run_telepathy(
        img_size:    int   = 32,
        n_iters_ai1: int   = 3000,
        n_iters_ai2: int   = 2000,
        n_anchor:    int   = 36,      # anchor points per phase
        noise_floor: float = 0.02,    # transmission noise σ
        out_path:    str   = "telepathy_result.png",
    ) -> dict:

    print("=" * 62)
    print("AI TELEPATHY — Prime-Wave Knowledge Transfer")
    print("=" * 62)

    # ── Step 1: Create images ──────────────────────────────────
    img_a = make_image_a(img_size)
    img_b = make_image_b(img_size)
    print(f"\n[1/6] Images created: {img_a.shape}")

    # ── Step 2: Train AI-1 ─────────────────────────────────────
    print(f"\n[2/6] Training AI-1 on both images ({n_iters_ai1} iters)…")
    t0   = time.time()
    ai1  = JanusNumpy(hidden=64, n_freqs=16, seed=0)
    ai1.train(img_a, img_b, n_iters=n_iters_ai1, verbose=True)
    print(f"  AI-1 training done in {time.time()-t0:.1f}s")

    # ── Step 3: AI-1 samples its knowledge ────────────────────
    print(f"\n[3/6] AI-1 samples knowledge ({n_anchor} pts × 2 phases)…")
    phases       = [0.0, np.pi/2]
    raw_vals, anchor_coords, phases = ai1.sample_knowledge(
        n_points=n_anchor, phases=phases
    )
    n_transmitted = len(raw_vals)
    print(f"  Payload: {n_transmitted} values "
          f"({n_anchor} pts × 2 phases × 3 ch)")

    # ── Step 4: Wave encode ─────────────────────────────────────
    print(f"\n[4/6] Encoding as prime-frequency wave…")
    # Split into two chunks (one per phase) for cleaner encoding
    chunk_size = n_anchor * 3   # n_points × 3 RGB channels
    codec      = WaveCodec(n_values=chunk_size,
                            base_freq=1.0, duration=60.0,
                            sample_rate=256,
                            noise_floor=noise_floor)

    chunks_a = raw_vals[:chunk_size]
    chunks_b = raw_vals[chunk_size:]

    wave_a = codec.encode(chunks_a)
    wave_b = codec.encode(chunks_b)
    print(f"  Codec: {chunk_size} values/phase  "
          f"cond={codec.condition_number:.2e}")

    # ── Step 5: Decode ──────────────────────────────────────────
    print(f"\n[5/6] Decoding received waves…")
    rec_a, fid_a = codec.decode(wave_a)
    rec_b, fid_b = codec.decode(wave_b)
    fidelity     = (fid_a + fid_b) / 2.0
    rec_vals     = np.concatenate([rec_a, rec_b])
    print(f"  Fidelity A: {fid_a:.2f}%  B: {fid_b:.2f}%  "
          f"mean: {fidelity:.2f}%")

    # ── Step 6: Train AI-2 on decoded samples ──────────────────
    print(f"\n[6/6] Training AI-2 from transmitted knowledge "
          f"({n_iters_ai2} iters)…")
    t0   = time.time()
    ai2  = JanusNumpy(hidden=64, n_freqs=16, seed=99)
    ai2.learn_from_samples(rec_vals, anchor_coords, phases,
                           n_iters=n_iters_ai2, verbose=True)
    print(f"  AI-2 training done in {time.time()-t0:.1f}s")

    # ── Evaluation ─────────────────────────────────────────────
    r_ai1_a = full_render(ai1, img_size, 0.0)
    r_ai1_b = full_render(ai1, img_size, np.pi/2)
    r_ai2_a = full_render(ai2, img_size, 0.0)
    r_ai2_b = full_render(ai2, img_size, np.pi/2)

    psnr_ai1_a = psnr(r_ai1_a, img_a)
    psnr_ai1_b = psnr(r_ai1_b, img_b)
    psnr_ai2_a = psnr(r_ai2_a, img_a)
    psnr_ai2_b = psnr(r_ai2_b, img_b)

    # AI-1 vs AI-2 agreement
    psnr_agree_a = psnr(r_ai2_a, r_ai1_a)
    psnr_agree_b = psnr(r_ai2_b, r_ai1_b)

    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    print(f"  AI-1 PSNR (vs originals): A={psnr_ai1_a:.1f} dB  B={psnr_ai1_b:.1f} dB")
    print(f"  AI-2 PSNR (vs originals): A={psnr_ai2_a:.1f} dB  B={psnr_ai2_b:.1f} dB")
    print(f"  AI-1 vs AI-2 agreement:   A={psnr_agree_a:.1f} dB  B={psnr_agree_b:.1f} dB")
    print(f"  Wave fidelity:            {fidelity:.2f}%")
    print(f"  Payload:                  {n_transmitted} values  "
          f"({n_transmitted*4/(1024):.1f} KB float32)")
    print(f"  Transmission time:        {codec.duration:.0f}s "
          f"@ {codec.sr} Hz = {len(codec.t)} samples")
    print("=" * 62)

    # ── Plot ────────────────────────────────────────────────────
    make_plots(img_a, img_b, ai1, ai2,
               wave_a, wave_a + np.random.randn(len(wave_a)) * noise_floor,
               codec, fidelity,
               ai1.loss_history, ai2.loss_history,
               out_path=out_path)

    return dict(
        psnr_ai1_a=psnr_ai1_a, psnr_ai1_b=psnr_ai1_b,
        psnr_ai2_a=psnr_ai2_a, psnr_ai2_b=psnr_ai2_b,
        agreement_a=psnr_agree_a, agreement_b=psnr_agree_b,
        fidelity=fidelity,
    )


# ─────────────────────────────────────────────────────────────────
# 5.  Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AI Telepathy demo")
    p.add_argument("--size",    type=int,   default=32,  help="image size (px)")
    p.add_argument("--iters1",  type=int,   default=3000, help="AI-1 training iters")
    p.add_argument("--iters2",  type=int,   default=2000, help="AI-2 training iters")
    p.add_argument("--anchor",  type=int,   default=36,  help="anchor points")
    p.add_argument("--noise",   type=float, default=0.02, help="channel noise σ")
    p.add_argument("--out",     type=str,   default="telepathy_result.png")
    args = p.parse_args()

    run_telepathy(
        img_size=args.size,
        n_iters_ai1=args.iters1,
        n_iters_ai2=args.iters2,
        n_anchor=args.anchor,
        noise_floor=args.noise,
        out_path=args.out,
    )
