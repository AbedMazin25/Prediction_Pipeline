#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(
        description="Load a .npy waveform (1‑D or 2×N) and plot PCG and/or ECG"
    )
    p.add_argument("--input", "-i", required=True,
                   help="path to your .npy file (1‑D or 2×N array)")
    p.add_argument("--fs",    "-f", type=float, default=1000,
                   help="sampling rate in Hz (default: 1000)")
    p.add_argument("--title", "-t", default=None,
                   help="overall figure title")
    args = p.parse_args()

    arr = np.load(args.input)
    # if 2×N assume [ECG; PCG]
    if arr.ndim == 2 and arr.shape[0] == 2:
        ecg = arr[0].flatten()
        pcg = arr[1].flatten()
        t = np.arange(ecg.size) / args.fs
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        if args.title:
            fig.suptitle(args.title)
        ax1.plot(t, ecg, color='tab:red', lw=1)
        ax1.set_title("ECG")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)

        ax2.plot(t, pcg, color='tab:blue', lw=1)
        ax2.set_title("PCG")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)

    else:
        # single trace
        data = arr.flatten()
        t = np.arange(data.size) / args.fs
        plt.figure(figsize=(10,4))
        if args.title:
            plt.title(args.title)
        plt.plot(t, data, lw=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
