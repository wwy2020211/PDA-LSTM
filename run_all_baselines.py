#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all PDA-LSTM comparison baselines on THE SAME random QLC sample.

Included:
1. Original data
2. Random bitline rearrangement
3. Random wordline rearrangement
4. Greedy wordline arrangement
5. TSP-style parallel simulated annealing
6. WBVM-style QLC proxy, CL64
7. WBVM-style QLC proxy, CL128
8. DVDS-style QLC proxy, CL64
9. DVDS-style QLC proxy, CL128

WBVM/DVDS are explicitly proxy implementations because the PDA-LSTM PDF
does not contain their exact TLC mapping/score tables.
"""

import argparse
import time
import torch

from common_lcm import (
    set_seed, resolve_device, generate_random_qlc, build_score_lut,
    score_order_direct, build_page_triple_score_cube, score_orders_from_cube
)
from baseline_random_bitline import random_bitline_search
from baseline_random_wordline import random_wordline_search
from baseline_greedy import greedy_search
from baseline_tsp_sa import parallel_simulated_annealing
from baseline_wbvm import wbvm_proxy
from baseline_dvds import dvds_proxy


def row(name, score, original, seconds):
    inc = score - original
    pct = 100.0 * inc / abs(original)
    return (name, score, inc, pct, seconds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pages", type=int, default=16)
    p.add_argument("--page-cells", type=int, default=18 * 1024 * 8)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--score-chunk", type=int, default=2048)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument("--random-bitline-trials", type=int, default=4)
    p.add_argument("--random-wordline-trials", type=int, default=4096)
    p.add_argument("--sa-iterations", type=int, default=3000)
    p.add_argument("--sa-chains", type=int, default=256)
    p.add_argument("--pe-cycles", type=int, default=500)
    args = p.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)

    x = generate_random_qlc(
        args.n_pages, args.page_cells, args.seed, device
    )
    lut = build_score_lut(device, alpha=args.alpha)

    original_t = time.perf_counter()
    original_score_t = score_order_direct(
        x, score_lut=lut, chunk_size=args.score_chunk
    )
    original_score = float(original_score_t)
    rows = [row(
        "Original data",
        original_score,
        original_score,
        time.perf_counter() - original_t
    )]

    # Precompute cube once for all wordline-order algorithms.
    t0 = time.perf_counter()
    cube = build_page_triple_score_cube(
        x, score_lut=lut, chunk_size=args.score_chunk
    )
    cube_time = time.perf_counter() - t0

    # Random wordline
    t0 = time.perf_counter()
    _, score = random_wordline_search(
        cube,
        trials=args.random_wordline_trials,
        batch=min(1024, args.random_wordline_trials),
        seed=args.seed + 11,
    )
    rows.append(row(
        "Random wordline",
        float(score),
        original_score,
        time.perf_counter() - t0 + cube_time
    ))

    # Greedy
    t0 = time.perf_counter()
    _, score = greedy_search(cube)
    rows.append(row(
        "Greedy",
        float(score),
        original_score,
        time.perf_counter() - t0 + cube_time
    ))

    # TSP / SA
    t0 = time.perf_counter()
    _, score = parallel_simulated_annealing(
        cube,
        iterations=args.sa_iterations,
        chains=args.sa_chains,
        seed=args.seed + 22,
    )
    rows.append(row(
        "TSP / SA",
        float(score),
        original_score,
        time.perf_counter() - t0 + cube_time
    ))

    # Random bitline
    t0 = time.perf_counter()
    _, score = random_bitline_search(
        x, lut,
        trials=args.random_bitline_trials,
        seed=args.seed + 33,
        chunk_size=args.score_chunk,
    )
    rows.append(row(
        "Random bitline",
        float(score),
        original_score,
        time.perf_counter() - t0
    ))

    # WBVM proxies
    for cl in (64, 128):
        t0 = time.perf_counter()
        y, _ = wbvm_proxy(x, code_length=cl)
        score = score_order_direct(
            y, score_lut=lut, chunk_size=args.score_chunk
        )
        rows.append(row(
            f"WBVM proxy CL{cl}",
            float(score),
            original_score,
            time.perf_counter() - t0
        ))

    # DVDS proxies
    for cl in (64, 128):
        t0 = time.perf_counter()
        y, _ = dvds_proxy(
            x,
            code_length=cl,
            pe_cycles=args.pe_cycles,
        )
        score = score_order_direct(
            y, score_lut=lut, chunk_size=args.score_chunk
        )
        rows.append(row(
            f"DVDS proxy CL{cl}",
            float(score),
            original_score,
            time.perf_counter() - t0
        ))

    print()
    print(f"device={device}, X={tuple(x.shape)}, seed={args.seed}")
    print("-" * 93)
    print(f"{'Algorithm':<24}{'Score':>19}{'Increment':>19}{'Increment %':>15}{'Time(s)':>15}")
    print("-" * 93)
    for name, score, inc, pct, sec in rows:
        print(f"{name:<24}{score:>19.6e}{inc:>19.6e}{pct:>14.6f}%{sec:>15.3f}")
    print("-" * 93)
    print("NOTE: WBVM/DVDS rows are QLC proxy implementations; see README.md.")


if __name__ == "__main__":
    main()
