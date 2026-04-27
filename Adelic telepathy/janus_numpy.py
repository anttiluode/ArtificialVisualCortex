"""
janus_numpy.py — Phase-Orthogonal Image Network (pure numpy)
=============================================================
A coordinate-based implicit neural representation that stores TWO
images in one set of weights by using phase-rotated input encoding.

Janus principle:
  - Image A is learned at phase φ = 0
  - Image B is learned at phase φ = π/2
  - The coordinate embedding is rotated by φ before entering the network
  - Because cos(0)=1, sin(0)=0 and cos(π/2)=0, sin(π/2)=1, the two
    input vectors are ORTHOGONAL → the network can satisfy both
    training constraints simultaneously

Architecture:
  (x, y) → Phase-rotated Fourier features → MLP → RGB

The "brain wave" = |Z(t)| where Z is the adelic sum over the weights
  projected onto prime-frequency oscillators.
"""

import numpy as np
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# Fourier feature embedding  (improves learning of fine detail)
# ─────────────────────────────────────────────────────────────────
class FourierEncoder:
    """
    Random Fourier Features for 2D coordinates.
    Maps (x,y) → [cos(Bx), sin(Bx)] ∈ R^{2K}
    with phase rotation applied BEFORE the frequency matrix.
    """
    def __init__(self, n_freqs: int = 16, scale: float = 6.0, seed: int = 0):
        rng = np.random.RandomState(seed)
        # Frequency matrix: shape (2, n_freqs)
        self.B = rng.randn(2, n_freqs) * scale
        self.out_dim = 2 * n_freqs

    def encode(self, coords: np.ndarray, phase: float = 0.0) -> np.ndarray:
        """
        coords : (N, 2)  — (x, y) in [-1, 1]
        phase  : scalar  — rotation angle in radians
        Returns: (N, 2*n_freqs)
        """
        c, s = np.cos(phase), np.sin(phase)
        # Rotate coordinates by phase angle
        x, y     = coords[:, 0], coords[:, 1]
        x_rot    = x * c - y * s
        y_rot    = x * s + y * c
        coords_r = np.stack([x_rot, y_rot], axis=1)  # (N, 2)

        proj = coords_r @ self.B    # (N, n_freqs)
        return np.concatenate([np.cos(proj), np.sin(proj)], axis=1)  # (N, 2K)


# ─────────────────────────────────────────────────────────────────
# Minimal Adam optimizer
# ─────────────────────────────────────────────────────────────────
class AdamOpt:
    def __init__(self, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for name, g in grads.items():
            if name not in self.m:
                self.m[name] = np.zeros_like(g)
                self.v[name] = np.zeros_like(g)
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * g**2
            m_hat = self.m[name] / (1 - self.b1**self.t)
            v_hat = self.v[name] / (1 - self.b2**self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ─────────────────────────────────────────────────────────────────
# The Janus Network
# ─────────────────────────────────────────────────────────────────
class JanusNumpy:
    """
    Stores two 2D→RGB images via phase-orthogonal coordinate encoding.

    Training:
        net.train(img_a, img_b, n_iters=2000)

    Inference:
        rgb = net.predict(coords, phase=0.0)      # Image A
        rgb = net.predict(coords, phase=np.pi/2)  # Image B
        rgb = net.predict(coords, phase=angle)    # Blend
    """

    def __init__(self,
                 hidden:   int   = 64,
                 n_freqs:  int   = 16,
                 ff_scale: float = 6.0,
                 seed:     int   = 42):

        rng  = np.random.RandomState(seed)
        self.encoder = FourierEncoder(n_freqs=n_freqs,
                                       scale=ff_scale, seed=seed)
        D = self.encoder.out_dim   # input dimension after encoding

        # Xavier init
        def w(fan_in, fan_out):
            return rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)

        self.params = {
            'W1': w(D,      hidden), 'b1': np.zeros(hidden),
            'W2': w(hidden, hidden), 'b2': np.zeros(hidden),
            'W3': w(hidden, hidden), 'b3': np.zeros(hidden),
            'W4': w(hidden, 3),      'b4': np.zeros(3),
        }
        self.opt = AdamOpt(lr=0.005)
        self.loss_history: list[float] = []

    # ── Forward pass ──────────────────────────────────────────
    def predict(self, coords: np.ndarray, phase: float = 0.0) -> np.ndarray:
        """
        coords : (N, 2) normalised to [-1, 1]
        returns: (N, 3) RGB in [0, 1]
        """
        p = self.params
        x = self.encoder.encode(coords, phase)
        h = np.tanh(x @ p['W1'] + p['b1'])
        h = np.tanh(h @ p['W2'] + p['b2'])
        h = np.tanh(h @ p['W3'] + p['b3'])
        z = h @ p['W4'] + p['b4']
        return 1.0 / (1.0 + np.exp(-z))   # sigmoid → [0,1]

    # ── Backward pass (single phase) ──────────────────────────
    def _forward_backward(self,
                          coords: np.ndarray,
                          phase:  float,
                          target: np.ndarray) -> tuple[float, dict]:
        p  = self.params
        N  = len(coords)
        x  = self.encoder.encode(coords, phase)

        # ── Forward ───────────────────────────────────────────
        z1 = x  @ p['W1'] + p['b1']
        a1 = np.tanh(z1)
        z2 = a1 @ p['W2'] + p['b2']
        a2 = np.tanh(z2)
        z3 = a2 @ p['W3'] + p['b3']
        a3 = np.tanh(z3)
        z4 = a3 @ p['W4'] + p['b4']
        pred = 1.0 / (1.0 + np.exp(-z4))

        loss = float(np.mean((pred - target)**2))

        # ── Backward ──────────────────────────────────────────
        d  = 2.0 / N * (pred - target) * pred * (1.0 - pred)  # sigmoid grad
        dW4 = a3.T @ d;  db4 = d.sum(0)
        d   = (d @ p['W4'].T) * (1.0 - a3**2)
        dW3 = a2.T @ d;  db3 = d.sum(0)
        d   = (d @ p['W3'].T) * (1.0 - a2**2)
        dW2 = a1.T @ d;  db2 = d.sum(0)
        d   = (d @ p['W2'].T) * (1.0 - a1**2)
        dW1 = x.T  @ d;  db1 = d.sum(0)

        grads = {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2,
                 'W3':dW3,'b3':db3,'W4':dW4,'b4':db4}
        return loss, grads

    # ── Training ──────────────────────────────────────────────
    def train(self,
              img_a:    np.ndarray,
              img_b:    np.ndarray,
              n_iters:  int   = 3000,
              batch:    int   = 1024,
              verbose:  bool  = True) -> None:
        """
        img_a, img_b : (H, W, 3) float32 in [0,1]
        """
        H, W, _ = img_a.shape
        ys, xs  = np.meshgrid(np.linspace(-1,1,H), np.linspace(-1,1,W),
                               indexing='ij')
        coords  = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        rgb_a   = img_a.reshape(-1, 3).astype(np.float32)
        rgb_b   = img_b.reshape(-1, 3).astype(np.float32)

        rng = np.random.RandomState(0)

        for i in range(n_iters):
            idx = rng.randint(0, len(coords), batch)
            bx  = coords[idx]

            loss_a, ga = self._forward_backward(bx, 0.0,      rgb_a[idx])
            loss_b, gb = self._forward_backward(bx, np.pi/2,  rgb_b[idx])

            # Jointly update on combined gradient
            combined = {k: ga[k] + gb[k] for k in ga}
            self.opt.step(self.params, combined)

            total_loss = loss_a + loss_b
            self.loss_history.append(total_loss)

            if verbose and (i % 500 == 0 or i == n_iters - 1):
                print(f"  iter {i:4d}/{n_iters}  loss={total_loss:.5f}")

    # ── Sample the network's "knowledge state" ─────────────────
    def sample_knowledge(self,
                         n_points: int = 32,
                         phases: Optional[list] = None,
                         seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample network outputs at a fixed grid of coordinates and phases.
        This is the "knowledge fingerprint" to be transmitted.

        Returns:
          values  : 1D float array — flattened samples (the payload)
          coords  : (n_points, 2) — sampling coordinates
          phases  : list of phase angles used
        """
        if phases is None:
            phases = [0.0, np.pi/2]

        rng = np.random.RandomState(seed)
        # Use a regular grid for reproducibility
        side  = int(np.ceil(np.sqrt(n_points)))
        g     = np.linspace(-0.9, 0.9, side)
        gx, gy = np.meshgrid(g, g)
        coords = np.stack([gx.ravel(), gy.ravel()], axis=1)[:n_points]

        all_vals = []
        for phi in phases:
            rgb = self.predict(coords, phase=phi)
            all_vals.append(rgb.ravel())

        return np.concatenate(all_vals), coords, phases

    # ── Train on received samples (AI-2 side) ─────────────────
    def learn_from_samples(self,
                           values:  np.ndarray,
                           coords:  np.ndarray,
                           phases:  list,
                           n_iters: int   = 2000,
                           verbose: bool  = True) -> None:
        """
        Train this network to match a set of received (decoded) samples.
        Used by AI-2 to absorb knowledge from AI-1's transmission.
        """
        n_pts  = len(coords)
        n_ph   = len(phases)
        values = values.reshape(n_ph, n_pts, 3)

        rng = np.random.RandomState(1)

        for i in range(n_iters):
            # Random phase each iteration
            ph_idx  = rng.randint(0, n_ph)
            phi     = phases[ph_idx]
            targets = values[ph_idx]

            idx = rng.randint(0, n_pts, min(n_pts, 128))
            loss, grads = self._forward_backward(
                coords[idx], phi, targets[idx]
            )
            self.opt.step(self.params, grads)
            self.loss_history.append(loss)

            if verbose and (i % 500 == 0 or i == n_iters - 1):
                print(f"  iter {i:4d}/{n_iters}  loss={loss:.5f}")

    # ── Weight → amplitude vector (for optional direct encoding) ─
    def weights_to_vector(self) -> np.ndarray:
        """Flatten all weights to a 1-D vector."""
        return np.concatenate([p.ravel() for p in self.params.values()])

    def vector_to_weights(self, v: np.ndarray) -> None:
        """Restore weights from a 1-D vector."""
        offset = 0
        for name, p in self.params.items():
            n = p.size
            self.params[name] = v[offset:offset+n].reshape(p.shape).copy()
            offset += n
