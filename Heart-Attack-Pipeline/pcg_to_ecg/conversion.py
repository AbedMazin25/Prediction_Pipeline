#!/usr/bin/env python3
"""
conversion.py

Turn a raw PCG .npy waveform (or a 2‑channel spectrogram) into a predicted ECG .npy or .mat waveform,
apply optional Soft filters + Savitzky–Golay smoothing, and pop up a quick plot of both PCG and ECG.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.io import savemat   # ← added
import os

inv_stft = torchaudio.transforms.InverseSpectrogram(n_fft=255, win_length=16)

# ── U‑Net building blocks ────────────────────────────────────────────────
class UnetDown1x2(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, normalize=True, dropout=0.0, prop=False):
        super().__init__()
        layers = []
        if prop:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        else:
            ks = (3 if stride==1 else 4, 4)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=ks,
                                    stride=(stride,2), padding=(1,1), bias=False))
        if normalize:     layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:       layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UnetUp1x2(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, dropout=0.0, normalize=True, prop=False):
        super().__init__()
        layers = []
        if prop:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1))
        else:
            ks = (3 if stride==1 else 4)
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, ks,
                                             stride=(stride,2), padding=1))
        if normalize:     layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:       layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, x_sk): return torch.cat([self.model(x), x_sk], dim=1)

class GeneraterUNet1x2_v4(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = UnetDown1x2(2,   8, normalize=False, prop=True)
        self.down1 = UnetDown1x2(8,  12)
        self.down2 = UnetDown1x2(12, 24)
        self.down3 = UnetDown1x2(24, 32)
        self.down4 = UnetDown1x2(32, 48, dropout=0.5, prop=True)
        self.down5 = UnetDown1x2(48, 48, dropout=0.5, prop=True)
        self.down6 = UnetDown1x2(48, 64, dropout=0.5)
        self.down7 = UnetDown1x2(64, 72, dropout=0.5, prop=True)
        self.down8 = UnetDown1x2(72, 84, dropout=0.5, prop=True)
        self.down9 = UnetDown1x2(84, 96, dropout=0.5)

        self.up2  = UnetUp1x2(96,   84, dropout=0.5)
        self.up3  = UnetUp1x2(84*2, 72, dropout=0.5, prop=True)
        self.up4  = UnetUp1x2(72*2, 64, dropout=0.5, prop=True)
        self.up5  = UnetUp1x2(64*2, 48, dropout=0.5)
        self.up6  = UnetUp1x2(48*2, 48, dropout=0.5, prop=True)
        self.up7  = UnetUp1x2(48*2, 32, dropout=0.5, prop=True)
        self.up8  = UnetUp1x2(32*2, 24, dropout=0.5)
        self.up9  = UnetUp1x2(24*2, 12)
        self.up10 = UnetUp1x2(12*2,  8)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(8*2, 2, kernel_size=(3,3), stride=(1,1), padding=1)
        )

    def forward(self, x):
        d0 = self.down0(x); d1 = self.down1(d0)
        d2 = self.down2(d1); d3 = self.down3(d2)
        d4 = self.down4(d3); d5 = self.down5(d4)
        d6 = self.down6(d5); d7 = self.down7(d6)
        d8 = self.down8(d7); z  = self.down9(d8)

        u2  = self.up2(z,  d8); u3  = self.up3(u2,  d7)
        u4  = self.up4(u3, d6); u5  = self.up5(u4,  d5)
        u6  = self.up6(u5, d4); u7  = self.up7(u6,  d3)
        u8  = self.up8(u7, d2); u9  = self.up9(u8,  d1)
        u10 = self.up10(u9, d0)

        return self.final(u10), None

# ── spectrogram → waveform inverse ──────────────────────────────────────────
def invert_spec(out_spec: torch.Tensor) -> torch.Tensor:
    mag = out_spec[:,0,:,3:-3]; ang = out_spec[:,1,:,3:-3]
    z   = torch.complex(mag*torch.cos(ang), mag*torch.sin(ang))
    return inv_stft(z.cpu())

def pcg_to_ecg(pcg: np.ndarray, gen: nn.Module, device: torch.device) -> np.ndarray:
    pcg = np.ascontiguousarray(pcg)
    # handle 2×N ECG/PCG segments
    if pcg.ndim==2 and pcg.shape[0]==2:
        pcg = pcg[1]  # second row is PCG
    # spectrogram path
    if pcg.ndim==3 and pcg.shape[0]==2:
        vsg = torch.from_numpy(pcg).unsqueeze(0).to(device)
        with torch.no_grad(): out_spec,_ = gen(vsg)
        return invert_spec(out_spec).squeeze().numpy()
    # raw waveform path
    pcg_t = torch.from_numpy(pcg).float()
    spec  = torchaudio.transforms.Spectrogram(win_length=16,n_fft=255,power=None)(pcg_t.unsqueeze(0))
    mag, ang = spec.abs(), spec.angle()
    mag, ang = F.pad(mag,(3,3)), F.pad(ang,(3,3))
    rem = mag.shape[-1] % 32
    if rem:
        mag, ang = F.pad(mag,(0,32-rem)), F.pad(ang,(0,32-rem))
    vsg = torch.cat([mag,ang],0).unsqueeze(0).to(device)
    with torch.no_grad(): out_spec,_ = gen(vsg)
    return invert_spec(out_spec).squeeze().numpy()

def convert_pcg_to_ecg(pcg: np.ndarray, weights_path=None, fs=1000, smooth_sec=0.05, polyorder=3) -> np.ndarray:
    """
    Convert PCG to ECG using the pre-trained model
    
    Parameters:
    -----------
    pcg : np.ndarray
        PCG signal
    weights_path : str, optional
        Path to model weights, if None will look for default path
    fs : float
        Sampling rate of PCG in Hz
    smooth_sec : float
        SavGol smoothing window length in seconds
    polyorder : int
        SavGol polynomial order
        
    Returns:
    --------
    ecg : np.ndarray
        Predicted ECG signal
    """
    # Set device with MPS support for M1 Macs
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Default weights path if not provided
    if weights_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, '..', 'models', 'enc_pixtopix_ecgppg_T7-V56.tm')
    
    # Load generator
    gen = GeneraterUNet1x2_v4().to(device)
    state = torch.load(weights_path, map_location=device)
    gen.load_state_dict(state)
    gen.eval()
    
    # Convert PCG to ECG
    ecg = pcg_to_ecg(pcg, gen, device)
    
    # Apply filters and smoothing
    nyq_e = 0.5 * fs
    b_e, a_e = butter(2, [0.1/nyq_e, 100/nyq_e], btype='band')
    ecg = filtfilt(b_e, a_e, ecg)
    win = int(smooth_sec * fs)
    if win % 2 == 0: win += 1
    ecg = savgol_filter(ecg, window_length=win, polyorder=polyorder)
    
    return ecg

# ── main ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    "-i", required=True,
                   help="raw PCG .npy or 2×N [ECG,PCG] segment or 2×F×T spec")
    p.add_argument("--weights",  "-w", required=True,
                   help="generator weights (.tm)")
    p.add_argument("--output",   "-o", required=True,
                   help="where to save ECG (.npy or .mat)")
    p.add_argument("--fs",       "-f", type=float, default=1000,
                   help="sampling rate of raw PCG")
    p.add_argument("--smooth_sec","-s", type=float, default=0.05,
                   help="SavGol smoothing window length in seconds")
    p.add_argument("--polyorder","-p", type=int, default=3,
                   help="SavGol polynomial order (must be < window samples)")
    p.add_argument("--no-plot",  action="store_true",
                   help="Do not display the plot")
    args = p.parse_args()

    # Updated device selection for M1 Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # load generator
    gen   = GeneraterUNet1x2_v4().to(device)
    state = torch.load(args.weights, map_location=device)
    gen.load_state_dict(state); gen.eval()

    seg = np.load(args.input)
    # extract PCG channel
    pcg = seg[1].copy() if (seg.ndim==2 and seg.shape[0]==2) else seg.copy()
    ecg = pcg_to_ecg(pcg, gen, device)

    # soften & smooth
    nyq_e = 0.5 * args.fs
    b_e,a_e = butter(2, [0.1/nyq_e,100/nyq_e], btype='band')
    ecg = filtfilt(b_e, a_e, ecg)
    win = int(args.smooth_sec * args.fs)
    if win % 2 == 0: win += 1
    ecg = savgol_filter(ecg, window_length=win, polyorder=args.polyorder)

    # save as .mat or .npy
    if args.output.lower().endswith(".mat"):
        savemat(args.output, {"val": ecg})
    else:
        np.save(args.output, ecg)
    print(f"Wrote {args.output}, shape={ecg.shape}")

    # plot both if not suppressed
    if not args.no_plot:
        t = np.arange(len(ecg)) / args.fs
        fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(10,6))
        ax1.plot(np.arange(len(pcg))/args.fs, pcg, lw=1)
        ax1.set_title("PCG (raw)"); ax1.set_ylabel("Amplitude"); ax1.grid(True)
        ax2.plot(t, ecg, lw=1, color='tab:red')
        ax2.set_title("Predicted ECG (smoothed)")
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Amplitude"); ax2.grid(True)
        plt.tight_layout(); plt.show()

if __name__=="__main__":
    main()



'''
COMMAND TO RUN MY SCRIPT - PCG -> ECG
python pcg_to_ecg/conversion.py \
  --input  pcg_to_ecg/testing_pcg/s_1000015.npy \
  --weights models/enc_pixtopix_ecgppg_T7-V56.tm \
  --output pcg_to_ecg/pred_ecg.npy \
  --fs     1000 \
  --smooth_sec 0.02 \
  --polyorder 1
'''

#.mat
