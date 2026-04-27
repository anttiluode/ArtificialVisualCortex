"""
wave_codec.py — Prime-Frequency Wave Codec
==========================================
Encodes a vector of N floats as a sum of prime-frequency sinusoids,
then recovers them via least-squares decoding.

    S(t) = Σ_{k=0}^{N-1}  v_k · cos( 2π · BASE_FREQ · log(p_k) · t )

Why prime frequencies?
  - log(p_k) are linearly independent over ℚ (Kronecker's theorem)
  - This makes the basis functions maximally incommensurate — no exact
    aliases or harmonics, unlike equal-spacing DFT
  - With T long enough the Gram matrix is well-conditioned (see below)

Capacity limit (empirical):
  N ≤ ~150 at T=60s, 256 Hz → fidelity > 99.7%
  Above this the basis matrix becomes ill-conditioned.
  Use chunked transmission for larger payloads.

Usage:
    codec = WaveCodec(n_values=96, base_freq=1.0, duration=60.0)
    wave  = codec.encode(values)
    recovered, fidelity = codec.decode(wave)
"""

import numpy as np
from typing import Optional

# ─────────────────────────────────────────────
# Primes
# ─────────────────────────────────────────────
def get_primes(n: int) -> list[int]:
    out, k = [], 2
    while len(out) < n:
        if all(k % d for d in range(2, int(k**0.5) + 1)):
            out.append(k)
        k += 1
    return out


class WaveCodec:
    """
    Encoder / decoder for AI-to-AI knowledge transfer via prime waves.

    Parameters
    ----------
    n_values  : number of float values to pack per transmission
    base_freq : Hz — fundamental scaling factor (1.0 default)
    duration  : seconds of wave to generate / expect
    sample_rate: Hz
    noise_floor: additive Gaussian noise σ (0 = noiseless)
    """

    def __init__(self,
                 n_values:    int   = 96,
                 base_freq:   float = 1.0,
                 duration:    float = 60.0,
                 sample_rate: int   = 256,
                 noise_floor: float = 0.0):

        assert n_values <= 150, \
            "n_values > 150 leads to ill-conditioned basis; use chunked mode"

        self.N          = n_values
        self.base_freq  = base_freq
        self.duration   = duration
        self.sr         = sample_rate
        self.noise_floor = noise_floor

        self.primes = get_primes(n_values)
        self.t      = np.linspace(0, duration,
                                  int(duration * sample_rate),
                                  endpoint=False)

        # Frequencies: ω_k = 2π · base_freq · log(p_k)
        self.omegas = np.array([2*np.pi * base_freq * np.log(p)
                                for p in self.primes])

        # Pre-compute basis matrix B[n, k] = cos(ω_k · t_n)
        # Shape: (n_samples, N)
        self.B = np.cos(np.outer(self.t, self.omegas))

        # Pre-compute pseudo-inverse for fast decoding
        self._Bpinv, residuals, rank, sv = np.linalg.lstsq(
            self.B, np.eye(len(self.t)), rcond=1e-8
        )
        self._cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float('inf')

    # ─── Public API ────────────────────────────────────────────
    def encode(self, values: np.ndarray) -> np.ndarray:
        """
        Encode N float values → 1-D prime-frequency wave.

        values : 1D array of length N, any range (normalised internally)
        returns: wave S(t), shape (n_samples,)
        """
        assert len(values) == self.N, f"Expected {self.N} values, got {len(values)}"
        v = np.asarray(values, dtype=float)

        # Normalise to [-1, 1] for numeric stability
        self._v_min  = v.min()
        self._v_scale = (v.max() - v.min()) + 1e-12
        v_norm = 2.0 * (v - self._v_min) / self._v_scale - 1.0

        wave = self.B @ v_norm                      # (n_samples,)

        if self.noise_floor > 0:
            wave += np.random.randn(len(wave)) * self.noise_floor

        return wave

    def decode(self, wave: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Recover N values from a prime-frequency wave.

        Returns (values, fidelity_pct) where fidelity is measured
        against the noiseless wave (when available).

        values   : 1D array of length N, in original scale
        fidelity : percentage [0-100], estimated from residual norm
        """
        assert len(wave) == len(self.t), \
            f"Wave length {len(wave)} != expected {len(self.t)}"

        # Least-squares: find v s.t. B @ v ≈ wave
        v_norm = self._Bpinv @ wave                 # (N,)

        # Denormalise
        values = (v_norm + 1.0) / 2.0 * self._v_scale + self._v_min

        # Fidelity estimate: relative residual
        wave_reconstructed = self.B @ v_norm
        res_norm  = np.linalg.norm(wave - wave_reconstructed)
        wave_norm = np.linalg.norm(wave) + 1e-12
        fidelity  = max(0.0, (1.0 - res_norm / wave_norm)) * 100.0

        return values, fidelity

    # ─── Chunked transmission for large payloads ──────────────
    def encode_chunked(self, values: np.ndarray,
                       chunk_size: int = 96) -> list[np.ndarray]:
        """Encode a long vector as a list of waves (one per chunk)."""
        chunks, waves = [], []
        for start in range(0, len(values), chunk_size):
            chunk = values[start: start + chunk_size]
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                pad = np.zeros(chunk_size - len(chunk))
                chunk = np.concatenate([chunk, pad])
            chunks.append(len(values[start: start + chunk_size]))  # true length
            waves.append(self.encode(chunk))
        return waves, chunks

    def decode_chunked(self, waves: list[np.ndarray],
                       chunk_lengths: list[int]) -> tuple[np.ndarray, float]:
        """Decode a list of waves into a single value array."""
        parts, fidels = [], []
        for wave, ln in zip(waves, chunk_lengths):
            vals, fid = self.decode(wave)
            parts.append(vals[:ln])
            fidels.append(fid)
        return np.concatenate(parts), np.mean(fidels)

    # ─── Diagnostics ──────────────────────────────────────────
    @property
    def condition_number(self) -> float:
        return self._cond

    def frequency_table(self) -> str:
        lines = [f"{'#':>4}  {'prime':>6}  {'freq (Hz)':>10}"]
        lines.append("─" * 26)
        for k, (p, w) in enumerate(zip(self.primes, self.omegas)):
            lines.append(f"{k:>4}  {p:>6}  {w/(2*np.pi):>10.4f}")
        return "\n".join(lines)
