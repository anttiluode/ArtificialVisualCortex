"""
JANUS CABBAGE: HOLOGRAPHIC SUPERPOSITION
========================================
A demonstration of Phase-Orthogonal Storage in Complex Neural Networks.

THE GOAL:
Store TWO different images in the EXACT SAME set of weights.

THE PHYSICS:
1. Reality A is stored at Phase 0 (Real Axis).
2. Reality B is stored at Phase PI/2 (Imaginary Axis).
3. Because the axes are orthogonal, the network can learn both perfectly without interference.
4. "The Slider" lets you rotate your viewing angle (Phase) to cross between worlds.

Usage:
1. Load Image 1 (The Day World).
2. Load Image 2 (The Night World).
3. Train.
4. Use the slider to rotate the phase and reveal the hidden layer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from threading import Thread
import time

# ============================================================================
# 1. THE HOLOGRAPHIC BRAIN (Identical to Cabbage3)
# ============================================================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(out_features))
        self.bias_i = nn.Parameter(torch.zeros(out_features))
        # Criticality Init
        nn.init.xavier_normal_(self.fc_r.weight, gain=0.2)
        nn.init.xavier_normal_(self.fc_i.weight, gain=0.2)

    def forward(self, z):
        # z = (real, imag)
        r, i = z[..., 0], z[..., 1]
        out_r = self.fc_r(r) - self.fc_i(i) + self.bias_r
        out_i = self.fc_r(i) + self.fc_i(r) + self.bias_i
        return torch.stack([out_r, out_i], dim=-1)

class JanusNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_scale = 10.0
        # Standard Cabbage Architecture
        self.l1 = ComplexLinear(2, 64)
        self.l2 = ComplexLinear(64, 64)
        self.l3 = ComplexLinear(64, 64)
        self.out = ComplexLinear(64, 3) # RGB

    def forward(self, x, phase_shift=0.0):
        # 1. Embed Coordinates
        # (x, y) -> Complex Plane
        z = torch.stack([x * np.pi * self.feature_scale, x * np.pi * self.feature_scale * 0.5], dim=-1)
        
        # 2. APPLY GLOBAL PHASE ROTATION (The "Viewing Angle")
        # z_new = z * e^(i * theta)
        # Real = r*cos - i*sin, Imag = r*sin + i*cos
        if phase_shift != 0.0:
            theta = torch.tensor(phase_shift, device=x.device)
            c, s = torch.cos(theta), torch.sin(theta)
            r, i = z[..., 0], z[..., 1]
            z_rot_r = r * c - i * s
            z_rot_i = r * s + i * c
            z = torch.stack([z_rot_r, z_rot_i], dim=-1)
        
        # 3. Resonant Processing
        z = self.l1(z); z = torch.tanh(z)
        z = self.l2(z); z = torch.tanh(z)
        z = self.l3(z); z = torch.tanh(z)
        
        # 4. Readout
        z_out = self.out(z)
        rgb = torch.sqrt(z_out[..., 0]**2 + z_out[..., 1]**2)
        return rgb

# ============================================================================
# 2. THE GUI
# ============================================================================
class JanusApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Janus Cabbage: Holographic Superposition")
        self.root.geometry("1000x600")
        self.root.configure(bg="#111")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = JanusNet().to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=0.005)
        
        self.img1_np = None
        self.img2_np = None
        self.is_training = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Controls
        panel = tk.Frame(self.root, bg="#222", width=250)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        btn_style = {"bg": "#444", "fg": "white", "font": ("Consolas", 10), "relief": "flat"}
        
        tk.Label(panel, text="REALITY A (Phase 0)", bg="#222", fg="#aaa").pack(pady=(10,0))
        tk.Button(panel, text="Load Image A", command=lambda: self.load_image(0), **btn_style).pack(pady=5, fill=tk.X)
        self.lbl_img1 = tk.Label(panel, bg="black", width=20, height=10); self.lbl_img1.pack()
        
        tk.Label(panel, text="REALITY B (Phase PI/2)", bg="#222", fg="#aaa").pack(pady=(20,0))
        tk.Button(panel, text="Load Image B", command=lambda: self.load_image(1), **btn_style).pack(pady=5, fill=tk.X)
        self.lbl_img2 = tk.Label(panel, bg="black", width=20, height=10); self.lbl_img2.pack()
        
        self.btn_train = tk.Button(panel, text="START SUPERPOSITION", command=self.toggle_train, bg="#005500", fg="white", font=("Consolas", 12, "bold"))
        self.btn_train.pack(pady=30, fill=tk.X)
        
        self.lbl_loss = tk.Label(panel, text="Waiting...", bg="#222", fg="#0f0")
        self.lbl_loss.pack()

        # Canvas & Slider
        right = tk.Frame(self.root, bg="#111")
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.lbl_view = tk.Label(right, bg="black")
        self.lbl_view.pack(expand=True)
        
        tk.Label(right, text="PHASE ROTATION", bg="#111", fg="white", font=("Consolas", 10)).pack(pady=5)
        self.slider = tk.Scale(right, from_=0, to=90, orient=tk.HORIZONTAL, bg="#222", fg="white", length=400, command=self.on_slide)
        self.slider.pack(pady=20)
        self.slider.set(0) # Start at Reality A

    def load_image(self, slot):
        path = filedialog.askopenfilename()
        if not path: return
        img = Image.open(path).convert('RGB').resize((256, 256))
        arr = np.array(img) / 255.0
        
        # Display thumbnail
        thumb = ImageTk.PhotoImage(img.resize((100, 100)))
        
        if slot == 0:
            self.img1_np = arr
            self.lbl_img1.config(image=thumb); self.lbl_img1.image = thumb
        else:
            self.img2_np = arr
            self.lbl_img2.config(image=thumb); self.lbl_img2.image = thumb
            
    def toggle_train(self):
        if self.img1_np is not None and self.img2_np is not None:
            self.is_training = not self.is_training
            if self.is_training:
                self.btn_train.config(text="STOP", bg="#550000")
                Thread(target=self.train_loop, daemon=True).start()
            else:
                self.btn_train.config(text="RESUME", bg="#005500")

    def train_loop(self):
        # Prepare Coordinates
        h, w, _ = self.img1_np.shape
        ys, xs = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
        coords = np.stack([xs, ys], axis=-1).astype(np.float32)
        coords = torch.tensor(coords).to(self.device).view(-1, 2)
        
        t1 = torch.tensor(self.img1_np, dtype=torch.float32).to(self.device).view(-1, 3)
        t2 = torch.tensor(self.img2_np, dtype=torch.float32).to(self.device).view(-1, 3)
        
        batch_size = 4096 * 4
        
        while self.is_training:
            # Batch Training
            idx = np.random.randint(0, len(coords), batch_size)
            batch_x = coords[idx]
            batch_y1 = t1[idx]
            batch_y2 = t2[idx]
            
            self.opt.zero_grad()
            
            # 1. Forward Pass Phase 0 (Reality A)
            pred1 = self.model(batch_x, phase_shift=0.0)
            loss1 = nn.MSELoss()(pred1, batch_y1)
            
            # 2. Forward Pass Phase 90 (Reality B)
            # We rotate the INPUT coordinates by 90 degrees (pi/2)
            pred2 = self.model(batch_x, phase_shift=np.pi/2)
            loss2 = nn.MSELoss()(pred2, batch_y2)
            
            # 3. Joint Loss
            # The network must satisfy BOTH constraints.
            # It must find a weight configuration where w*x = A AND w*(ix) = B
            loss = loss1 + loss2
            loss.backward()
            self.opt.step()
            
            if np.random.rand() < 0.05:
                self.update_stats(loss.item())

    def update_stats(self, loss):
        self.lbl_loss.config(text=f"Joint Loss: {loss:.5f}")
        # Trigger view update based on current slider position
        self.on_slide(self.slider.get())

    def on_slide(self, val):
        if not self.is_training and self.img1_np is None: return
        
        deg = float(val)
        rad = np.radians(deg) # Convert 0-90 to 0-PI/2
        
        # Render Full View
        with torch.no_grad():
            # Create low-res preview for speed
            res = 128
            ys, xs = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res), indexing='ij')
            coords = np.stack([xs, ys], axis=-1).astype(np.float32)
            coords_t = torch.tensor(coords).to(self.device)
            
            # INFERENCE AT CURRENT PHASE ANGLE
            pred = self.model(coords_t, phase_shift=rad)
            
            img = pred.cpu().numpy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
            
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.lbl_view.config(image=img_tk)
            self.lbl_view.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = JanusApp(root)
    root.mainloop()