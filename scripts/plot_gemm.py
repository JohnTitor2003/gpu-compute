#!/usr/bin/env python3
"""Roofline + speedup charts from gemm_bench --csv output."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# RTX 5080 (GB203, sm_120): 10752 FP32 cores, 2.617 GHz boost, 960 GB/s.
PEAK_TFLOPS = 10752 * 2 * 2.617 / 1e3  # FMA = 2 FLOP
PEAK_BW_GBS = 960.0
RIDGE = PEAK_TFLOPS * 1e3 / PEAK_BW_GBS  # FLOP/byte


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["M"] = int(r["M"])
            r["N"] = int(r["N"])
            r["K"] = int(r["K"])
            for k in (
                "ai",
                "auto_ms",
                "cublas_fp32_ms",
                "cublas_tf32_ms",
                "auto_tflops",
                "cublas_fp32_tflops",
                "cublas_tf32_tflops",
                "speedup_fp32",
            ):
                r[k] = float(r[k])
            r["win"] = int(r["win"])
            rows.append(r)
    return rows


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
        }
    )


def plot_speedup(rows: list[dict], family: str, title: str, out: Path, xlabel: str):
    data = [r for r in rows if r["family"] == family]
    if family == "skinny":
        # one M=K family for a clean N sweep — prefer 4096
        pref = [r for r in data if r["M"] == 4096]
        if pref:
            data = pref
    if not data:
        return
    data = sorted(data, key=lambda r: r["N"] if family == "skinny" else r["M"])
    labels = [r["shape"] if family == "square" else str(r["N"]) for r in data]
    sp = [r["speedup_fp32"] for r in data]
    colors = ["#0f7b3c" if s >= 1.0 else "#9b1c1c" for s in sp]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(range(len(sp)), sp, color=colors, width=0.72)
    ax.axhline(1.0, color="black", lw=1, ls="--", label="cuBLAS FP32")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("speedup vs pedantic cuBLAS FP32")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(sp):
        ax.text(i, v + 0.04, f"{v:.2f}×", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(2.0, max(sp) * 1.25))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_roofline(rows: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ai = np.logspace(-0.3, 3.1, 400)
    roof = np.minimum(PEAK_BW_GBS * ai / 1e3, PEAK_TFLOPS)
    ax.plot(ai, roof, color="#222", lw=1.6, label="RTX 5080 roofline (FP32 / HBM)")
    ax.axvline(RIDGE, color="#888", ls=":", lw=1)
    ax.text(RIDGE * 1.05, PEAK_TFLOPS * 0.55, f"ridge {RIDGE:.0f} F/B", rotation=90, va="center", fontsize=8, color="#555")

    def scatter(fam, marker, face, edge, label):
        xs, ys = [], []
        for r in rows:
            if r["family"] != fam:
                continue
            xs.append(r["ai"])
            ys.append(r["auto_tflops"])
        if xs:
            ax.scatter(xs, ys, s=42, marker=marker, facecolor=face, edgecolor=edge, lw=0.8, zorder=3, label=label)

    scatter("skinny", "o", "#1f6feb", "#0b3d91", "auto skinny")
    scatter("square", "s", "#e8590c", "#9f3a00", "auto square")

    cx, cy = [], []
    for r in rows:
        cx.append(r["ai"])
        cy.append(r["cublas_fp32_tflops"])
    ax.scatter(cx, cy, s=22, marker="x", color="#888", zorder=2, label="cuBLAS FP32")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("TFLOP/s")
    ax.set_title("SGEMM roofline — RTX 5080 Blackwell")
    ax.set_xlim(0.5, 1400)
    ax.set_ylim(0.02, PEAK_TFLOPS * 1.4)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_us(rows: list[dict], out: Path):
    data = [r for r in rows if r["family"] == "skinny" and r["M"] == 4096]
    data = sorted(data, key=lambda r: r["N"])
    if not data:
        return
    n = [r["N"] for r in data]
    ours = [r["auto_ms"] * 1e3 for r in data]
    cublas = [r["cublas_fp32_ms"] * 1e3 for r in data]
    x = np.arange(len(n))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.18, ours, 0.36, color="#1f6feb", label="this kernel")
    ax.bar(x + 0.18, cublas, 0.36, color="#adb5bd", label="cuBLAS FP32")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in n])
    ax.set_xlabel("N  (M = K = 4096)")
    ax.set_ylabel("microseconds")
    ax.set_title("Wall-clock kernel time — skinny SGEMM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gemm.csv")
    outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load(src)
    style()
    plot_speedup(
        rows,
        "skinny",
        "Speedup vs cuBLAS FP32 — skinny SGEMM (M=K=4096)",
        outdir / "speedup_skinny.png",
        "N",
    )
    plot_speedup(
        rows,
        "square",
        "Speedup vs cuBLAS FP32 — square SGEMM",
        outdir / "speedup_square.png",
        "M = N = K",
    )
    plot_roofline(rows, outdir / "roofline.png")
    plot_us(rows, outdir / "time_skinny.png")
    wins = sum(r["win"] for r in rows)
    print(f"wrote charts to {outdir}  ({wins}/{len(rows)} wins)")


if __name__ == "__main__":
    main()
