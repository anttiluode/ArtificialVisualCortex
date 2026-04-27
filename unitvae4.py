
import argparse
import os
import sys
import types
import threading
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image, ImageTk, ImageEnhance
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time

# ===============================
# Environment Setup & Monkey-Patch for Triton
# ===============================
os.environ["DIFFUSERS_NO_IP_ADAPTER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import triton.runtime
except ImportError:
    sys.modules["triton"] = types.ModuleType("triton")
    sys.modules["triton.runtime"] = types.ModuleType("triton.runtime")
    import triton.runtime

if not hasattr(triton.runtime, "Autotuner"):
    class DummyAutotuner:
        def __init__(self, *args, **kwargs):
            pass
        def tune(self, *args, **kwargs):
            return None
    triton.runtime.Autotuner = DummyAutotuner

# ===============================
# Imports from diffusers and other packages
# ===============================
from diffusers import StableVideoDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Define Adaptive VAE Components (Convolutional Encoder/Decoder)
# ===============================
class AdaptiveEncoderConv(nn.Module):
    def __init__(self):
        super(AdaptiveEncoderConv, self).__init__()
        # Downsample: 512x512 -> 256 -> 128 -> 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)    # 512 -> 256
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)   # 256 -> 128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        self.conv4 = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)      # keep 64x64, output channels=4
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        latent = self.conv4(x)  # Expected shape: [batch, 4, 64, 64]
        return latent

class AdaptiveDecoderConv(nn.Module):
    def __init__(self):
        super(AdaptiveDecoderConv, self).__init__()
        # Upsample: 64x64 -> 128x128 -> 256x256 -> 512x512.
        self.conv_trans1 = nn.ConvTranspose2d(4, 256, kernel_size=3, stride=1, padding=1)   # 64 remains
        self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)   # 64 -> 128
        self.conv_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)    # 128 -> 256
        self.conv_trans4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)      # 256 -> 512
        self.relu = nn.ReLU()
    
    def forward(self, latent):
        x = self.relu(self.conv_trans1(latent))
        x = self.relu(self.conv_trans2(x))
        x = self.relu(self.conv_trans3(x))
        recon = torch.sigmoid(self.conv_trans4(x))  # Output in [0,1]
        return recon

# ===============================
# AdaptiveVAETrainer: Performs a training step on a single frame.
# ===============================
class AdaptiveVAETrainer:
    def __init__(self, encoder, decoder, teacher_vae):
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_vae = teacher_vae  # teacher_vae from diffusers pipeline (pipe.vae)
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_on_frame(self, image_tensor):
        # image_tensor: FP32 tensor [1, 3, 512, 512]
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        # Teacher VAE expects FP16: convert input accordingly and then convert outputs back to FP32.
        with torch.no_grad():
            teacher_latent = self.teacher_vae.encode(image_tensor.half()).latent_dist.sample().float()
            # Note: Pass num_frames=1 to decode.
            decoded = self.teacher_vae.decode(teacher_latent.half(), num_frames=1).sample
            teacher_decoded = ((decoded / 2 + 0.5).clamp(0, 1)).float()
        # Use AMP autocast (without device_type keyword for compatibility)
        with torch.cuda.amp.autocast():
            pred_latent = self.encoder(image_tensor)
            latent_loss = self.loss_fn(pred_latent, teacher_latent)
            pred_image = self.decoder(pred_latent)
            image_loss = self.loss_fn(pred_image, teacher_decoded)
            loss = latent_loss + image_loss
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

# ===============================
# LatentVideoFilter: GUI with Webcam, Teach Mode, and Adaptive VAE integration
# ===============================
class LatentVideoFilter:
    def __init__(self, master):
        self.master = master
        self.master.title("Live Webcam - Adaptive VAE")
        self.device = device
        
        # Load the teacher VAE using StableVideoDiffusionPipeline (for img2vid-style encoding/decoding)
        print("Loading Stable Video Diffusion (img2vid-xt)...")
        self.video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Setup transformation for 512x512 images.
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        
        # Initialize video capture variables
        self.cap = None
        self.camera_index = 0
        self.available_cameras = self.get_camera_indices()
        
        # Set up GUI controls
        self.setup_gui()
        
        # Initialize the adaptive VAE (encoder and decoder in FP32)
        self.adaptive_encoder = AdaptiveEncoderConv().to(self.device)
        self.adaptive_decoder = AdaptiveDecoderConv().to(self.device)
        
        # Create the adaptive VAE trainer (using the teacher VAE from the pipeline)
        self.adaptive_trainer = AdaptiveVAETrainer(self.adaptive_encoder, self.adaptive_decoder, self.video_pipe.vae)
        
        # Teach mode flag and frame storage
        self.teach_mode = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Start a background thread for continuous training if teach mode is active
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()
        
        # Start updating the video feed
        self.update_video()
    
    def setup_gui(self):
        control_frame = tk.Frame(self.master)
        control_frame.pack(side='top', fill='x', padx=10, pady=5)
        
        # Teach Mode toggle button
        self.teach_button = tk.Button(control_frame, text="Start Teach Mode", command=self.toggle_teach_mode)
        self.teach_button.pack(side='left', padx=5)
        
        # Save Model button
        self.save_button = tk.Button(control_frame, text="Save Model", command=self.save_model)
        self.save_button.pack(side='left', padx=5)
        
        # Load Model button
        self.load_button = tk.Button(control_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(side='left', padx=5)
        
        # Video display label
        self.video_label = tk.Label(self.master)
        self.video_label.pack(padx=10, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.master, textvariable=self.status_var, relief='sunken', anchor='w')
        self.status_label.pack(side='bottom', fill='x')
    
    def get_camera_indices(self):
        indices = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                indices.append(f"Camera {i}")
                cap.release()
        return indices
    
    def toggle_teach_mode(self):
        self.teach_mode = not self.teach_mode
        if self.teach_mode:
            self.teach_button.config(text="Stop Teach Mode")
            self.status_var.set("Teach mode active")
        else:
            self.teach_button.config(text="Start Teach Mode")
            self.status_var.set("Teach mode paused")
    
    def save_model(self):
        filename = filedialog.asksaveasfilename(title="Save Adaptive VAE", defaultextension=".pth",
                                                  filetypes=[("PyTorch files", "*.pth")])
        if filename:
            torch.save({
                'encoder': self.adaptive_encoder.state_dict(),
                'decoder': self.adaptive_decoder.state_dict(),
            }, filename)
            self.status_var.set(f"Model saved to {filename}")
    
    def load_model(self):
        filename = filedialog.askopenfilename(title="Load Adaptive VAE",
                                              filetypes=[("PyTorch files", "*.pth")])
        if filename:
            checkpoint = torch.load(filename, map_location=self.device)
            self.adaptive_encoder.load_state_dict(checkpoint['encoder'])
            self.adaptive_decoder.load_state_dict(checkpoint['decoder'])
            self.status_var.set(f"Model loaded from {filename}")
    
    def training_loop(self):
        # Continuously train on the latest frame when teach mode is active.
        while True:
            if self.teach_mode and self.latest_frame is not None:
                with self.frame_lock:
                    frame = self.latest_frame.copy()
                try:
                    # Convert frame (BGR) to PIL image, then to tensor in FP32
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    transform = T.Compose([
                        T.Resize((512, 512)),
                        T.ToTensor(),
                    ])
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    loss = self.adaptive_trainer.train_on_frame(image_tensor)
                    self.status_var.set(f"Teach mode active, Loss: {loss:.4f}")
                except Exception as e:
                    self.status_var.set(f"Training error: {e}")
            time.sleep(0.1)  # Adjust training frequency as desired
    
    def update_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                # Process the frame through the adaptive VAE for reconstruction.
                try:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    transform = T.Compose([
                        T.Resize((512, 512)),
                        T.ToTensor(),
                    ])
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        latent = self.adaptive_encoder(image_tensor)
                        recon = self.adaptive_decoder(latent)
                    recon_np = recon.cpu().squeeze(0).permute(1, 2, 0).numpy()
                    recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
                    display_frame = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    self.status_var.set(f"Processing error: {e}")
                    display_frame = frame
                image_pil = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image=image_pil)
                self.video_label.config(image=photo)
                self.video_label.image = photo
            self.master.after(30, self.update_video)
    
    def run(self):
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.mainloop()
    
    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.master.destroy()

def main():
    root = tk.Tk()
    app = LatentVideoFilter(root)
    app.run()

if __name__ == "__main__":
    main()
