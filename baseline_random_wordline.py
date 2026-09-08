#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random wordline sorting baseline.

The PDA-LSTM paper says it randomly changes wordline ordering and observes
that many random iterations are needed to approach a high score.

This script generates many random wordline permutations in parallel and
selects the best one. It does NOT enumerate N!.
"""

import torch
from common_lcm import (
    make_common_parser, resolve_device, set_seed, generate_random_qlc,
    build_score_lut, build_page_triple_score_cube,
    score_order_direct, score_orders_from_cube, print_result
)


@torch.no_grad()
def random_wordline_search(s_ac, trials=4096, batch=1024, seed=2025):
    device = s_ac.device
    n = s_ac.shape[0]

    # torch.randperm cannot directly create M independent permutations,
    # so argsort of random keys creates them in a vectorized way.
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    best_score = torch.tensor(float("-inf"), device=device)
    best_order = None

    done = 0
    while done < trials:
        m = min(batch, trials - done)
        keys = torch.rand((m, n), generator=g, device=device)
        orders = torch.argsort(keys, dim=1)
        scores = score_orders_from_cube(s_ac, orders)

        idx = torch.argmax(scores)
        if scores[idx] > best_score:
            best_score = scores[idx]
            best_order = orders[idx].clone()
        done += m

    return best_order, best_score


def main():
    p = make_common_parser("Random wordline sorting baseline")
    p.add_argument("--trials", type=int, default=4096)
    p.add_argument("--trial-batch", type=int, default=1024)
    args = p.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)
    x = generate_random_qlc(args.n_pages, args.page_cells, args.seed, device)
    lut = build_score_lut(device, alpha=args.alpha)

    original = score_order_direct(x, score_lut=lut, chunk_size=args.score_chunk)
    cube = build_page_triple_score_cube(
        x, score_lut=lut, chunk_size=args.score_chunk
    )

    order, score = random_wordline_search(
        cube, trials=args.trials, batch=args.trial_batch, seed=args.seed + 1
    )

    print_result("Random wordline", float(original), float(score))
    print("best order      :", order.detach().cpu().tolist())


if __name__ == "__main__":
    main()
