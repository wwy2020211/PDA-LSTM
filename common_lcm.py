#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for PDA-LSTM baseline reproduction.

Data convention:
    X.shape == [N, D]
    N = number of physical pages / wordlines (paper uses 16)
    D = cells per physical page (paper uses 18*1024*8 = 147456)
    X values are QLC states 0..15.

The LCM score follows Eq.(1)-(4) and Table I of the PDA-LSTM paper.

Important:
- alpha is not numerically specified in the paper. Default alpha=1.0.
- k1:k2 = 4:1 per the paper.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class CommonConfig:
    n_pages: int = 16
    page_cells: int = 18 * 1024 * 8
    seed: int = 2025
    alpha: float = 1.0
    k1: float = 4.0
    k2: float = 1.0
    score_chunk: int = 2048


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_random_qlc(
    n_pages: int = 16,
    page_cells: int = 18 * 1024 * 8,
    seed: int = 2025,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Random QLC state data X[N,D] with values 0..15."""
    if device is None:
        device = default_device()

    # Generate on CPU for deterministic behavior, then move.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randint(
        0, 16,
        (n_pages, page_cells),
        generator=g,
        dtype=torch.uint8,
    )
    return x.to(device)


def build_score_lut(
    device: Optional[torch.device] = None,
    alpha: float = 1.0,
    k1: float = 4.0,
    k2: float = 1.0,
) -> torch.Tensor:
    """
    LUT[a,b,c] = s(a,b,c), shape [16,16,16].

    Ae rule from Table I:
      center E(0)             -> 5
      center P, E-P-E         -> 1
      center P, one neigh E   -> 2
      center P, P-P-P         -> 5
    """
    if device is None:
        device = default_device()
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    q = torch.arange(16, device=device, dtype=torch.float32)
    xa = q[:, None, None]
    xb = q[None, :, None]
    xc = q[None, None, :]

    base = xa + xb + xc
    center_e = xb == 0
    left_e = xa == 0
    right_e = xc == 0

    ae = torch.where(
        center_e,
        torch.full_like(base, 5.0),
        torch.where(
            left_e & right_e,
            torch.full_like(base, 1.0),
            torch.where(
                left_e ^ right_e,
                torch.full_like(base, 2.0),
                torch.full_like(base, 5.0),
            ),
        ),
    )

    f = (
        k2 * (16.0 - torch.abs(xa - xb))
        + k1 * (16.0 - torch.abs(xb - xc))
    ) / (alpha * (k1 + k2))

    return (ae * (16.0 - xb) * f).contiguous()


@torch.no_grad()
def score_order_direct(
    x: torch.Tensor,
    order: Optional[torch.Tensor] = None,
    score_lut: Optional[torch.Tensor] = None,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """
    Fast total score of ONE wordline order without N^3 cube.

    x: [N,D]
    order: position -> page index, [N]. If None, identity.

    score = sum_{t=0}^{N-3} sum_d LUT[
        X[order[t],d],
        X[order[t+1],d],
        X[order[t+2],d]
    ]

    Complexity O(N*D), no permutation enumeration.
    """
    if x.ndim != 2:
        raise ValueError("x must have shape [N,D]")

    device = x.device
    n, d = x.shape

    if score_lut is None:
        score_lut = build_score_lut(device=device)

    if order is None:
        y = x
    else:
        y = x[order.long()]

    lut_flat = score_lut.reshape(-1)
    total = torch.zeros((), device=device, dtype=torch.float64)

    for d0 in range(0, d, chunk_size):
        d1 = min(d0 + chunk_size, d)
        z = y[:, d0:d1].long()

        a = z[:-2]
        b = z[1:-1]
        c = z[2:]

        idx = a * 256 + b * 16 + c
        total += lut_flat[idx].double().sum()

    return total


@torch.no_grad()
def build_page_triple_score_cube(
    x: torch.Tensor,
    score_lut: Optional[torch.Tensor] = None,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Build S_AC[i,j,k] for every ordered page triple.

    x: [N,D]
    output: [N,N,N]

    Used by wordline-permutation baselines:
      random wordline, greedy, TSP/SA.

    Only N^3=4096 combinations are represented, not N!.
    Cell dimension is chunked to avoid creating [N,N,N,D] at once.
    """
    if x.ndim != 2:
        raise ValueError("x must have shape [N,D]")

    device = x.device
    n, d = x.shape

    if score_lut is None:
        score_lut = build_score_lut(device=device)

    lut_flat = score_lut.reshape(-1)
    out = torch.zeros((n, n, n), device=device, dtype=torch.float32)

    for d0 in range(0, d, chunk_size):
        d1 = min(d0 + chunk_size, d)
        z = x[:, d0:d1].long()          # [N,C]

        a = z[:, None, None, :]          # [N,1,1,C]
        b = z[None, :, None, :]          # [1,N,1,C]
        c = z[None, None, :, :]          # [1,1,N,C]

        idx = a * 256 + b * 16 + c       # [N,N,N,C]
        out += lut_flat[idx].sum(dim=-1)

    ids = torch.arange(n, device=device)
    i = ids[:, None, None]
    j = ids[None, :, None]
    k = ids[None, None, :]
    valid = (i != j) & (i != k) & (j != k)
    out *= valid.to(out.dtype)

    return out


@torch.no_grad()
def score_orders_from_cube(
    s_ac: torch.Tensor,
    orders: torch.Tensor,
) -> torch.Tensor:
    """
    Score many page permutations in parallel.

    s_ac: [N,N,N]
    orders: [M,N], each row position->page

    return scores [M]
    """
    if orders.ndim == 1:
        orders = orders.unsqueeze(0)

    a = orders[:, :-2].long()
    b = orders[:, 1:-1].long()
    c = orders[:, 2:].long()
    return s_ac[a, b, c].sum(dim=1)


def make_common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--n-pages", type=int, default=16)
    p.add_argument("--page-cells", type=int, default=18 * 1024 * 8)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--score-chunk", type=int, default=2048)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return p


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return default_device()
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def print_result(name: str, original_score: float, new_score: float) -> None:
    delta = new_score - original_score
    pct = 100.0 * delta / abs(original_score) if original_score != 0 else float("nan")
    print(f"algorithm       : {name}")
    print(f"original score  : {original_score:.6e}")
    print(f"result score    : {new_score:.6e}")
    print(f"score increment : {delta:.6e} ({pct:.6f}%)")
