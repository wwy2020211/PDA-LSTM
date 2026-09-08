#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSP-style simulated annealing baseline.

The PDA-LSTM paper:
- treats page arrangement as TSP-like sequence generation,
- discusses simulated annealing (SA) as a classic TSP solver,
- states that TSP repeatedly calculates LCM score during search.

The paper does not specify the exact TSP solver used in its experiment.
Therefore this file implements a clearly-labeled SA-TSP reproduction:
parallel independent SA chains on page permutations, scored by the paper's
full three-page LCM objective. This is NOT exhaustive permutation search.
"""

import math
import torch
from common_lcm import (
    make_common_parser, resolve_device, set_seed, generate_random_qlc,
    build_score_lut, build_page_triple_score_cube,
    score_order_direct, score_orders_from_cube, print_result
)


@torch.no_grad()
def parallel_simulated_annealing(
    s_ac,
    iterations=5000,
    chains=256,
    t0=1.0,
    t_end=1e-3,
    seed=2025,
):
    device = s_ac.device
    n = s_ac.shape[0]

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Random initial permutations in parallel.
    orders = torch.argsort(
        torch.rand((chains, n), generator=g, device=device),
        dim=1,
    )
    scores = score_orders_from_cube(s_ac, orders)

    best_idx = torch.argmax(scores)
    best_order = orders[best_idx].clone()
    best_score = scores[best_idx].clone()

    rows = torch.arange(chains, device=device)

    # Normalize energy scale so temperature is dimensionless and stable.
    scale = torch.clamp(scores.abs().mean(), min=1.0)

    for it in range(iterations):
        frac = it / max(iterations - 1, 1)
        temp = t0 * ((t_end / t0) ** frac)

        i = torch.randint(0, n, (chains,), generator=g, device=device)
        j = torch.randint(0, n, (chains,), generator=g, device=device)
        same = i == j
        j[same] = (j[same] + 1) % n

        proposal = orders.clone()
        vi = proposal[rows, i].clone()
        vj = proposal[rows, j].clone()
        proposal[rows, i] = vj
        proposal[rows, j] = vi

        new_scores = score_orders_from_cube(s_ac, proposal)
        delta = (new_scores - scores) / scale

        accept_prob = torch.exp(torch.clamp(delta / temp, max=0.0))
        rnd = torch.rand((chains,), generator=g, device=device)
        accept = (delta >= 0) | (rnd < accept_prob)

        orders[accept] = proposal[accept]
        scores[accept] = new_scores[accept]

        idx = torch.argmax(scores)
        if scores[idx] > best_score:
            best_score = scores[idx].clone()
            best_order = orders[idx].clone()

    return best_order, best_score


def main():
    p = make_common_parser("TSP simulated annealing baseline")
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--chains", type=int, default=256)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--t-end", type=float, default=1e-3)
    args = p.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)

    x = generate_random_qlc(args.n_pages, args.page_cells, args.seed, device)
    lut = build_score_lut(device, alpha=args.alpha)

    original = score_order_direct(x, score_lut=lut, chunk_size=args.score_chunk)
    cube = build_page_triple_score_cube(
        x, score_lut=lut, chunk_size=args.score_chunk
    )

    order, score = parallel_simulated_annealing(
        cube,
        iterations=args.iterations,
        chains=args.chains,
        t0=args.t0,
        t_end=args.t_end,
        seed=args.seed + 1,
    )

    print_result("TSP / Simulated Annealing", float(original), float(score))
    print("best order      :", order.detach().cpu().tolist())


if __name__ == "__main__":
    main()
