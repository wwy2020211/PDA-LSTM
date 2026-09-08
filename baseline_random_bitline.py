#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random bitline-direction rearrangement baseline.

The paper compares random movement in bitline and wordline directions.
To actually change vertical cell neighborhoods, each physical page receives
an independent random permutation of its cell/bitline positions.

For full 18KB pages this is expensive, so the default number of trials is small.
"""

import torch
from common_lcm import (
    make_common_parser, resolve_device, set_seed, generate_random_qlc,
    build_score_lut, score_order_direct, print_result
)


@torch.no_grad()
def random_bitline_search(x, score_lut, trials=8, seed=2025, chunk_size=8192):
    device = x.device
    n, d = x.shape

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    best_score = torch.tensor(float("-inf"), device=device, dtype=torch.float64)
    best_x = None

    for _ in range(trials):
        # Independent random cell permutation for each page.
        # argsort(random keys) is vectorized across pages.
        keys = torch.rand((n, d), generator=g, device=device)
        perm = torch.argsort(keys, dim=1)
        y = torch.gather(x, dim=1, index=perm)

        score = score_order_direct(
            y, score_lut=score_lut, chunk_size=chunk_size
        )
        if score > best_score:
            best_score = score
            best_x = y.clone()

    return best_x, best_score


def main():
    p = make_common_parser("Random bitline sorting baseline")
    p.add_argument("--trials", type=int, default=8)
    args = p.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)
    x = generate_random_qlc(args.n_pages, args.page_cells, args.seed, device)
    lut = build_score_lut(device, alpha=args.alpha)

    original = score_order_direct(x, score_lut=lut, chunk_size=args.score_chunk)
    _, score = random_bitline_search(
        x, lut, trials=args.trials, seed=args.seed + 1,
        chunk_size=args.score_chunk
    )

    print_result("Random bitline", float(original), float(score))


if __name__ == "__main__":
    main()
