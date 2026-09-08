#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Original-data baseline: no processing."""

import torch
from common_lcm import (
    make_common_parser, resolve_device, set_seed, generate_random_qlc,
    build_score_lut, score_order_direct, print_result
)


def original_baseline(x, score_lut, chunk_size=8192):
    score = score_order_direct(
        x, order=None, score_lut=score_lut, chunk_size=chunk_size
    )
    order = torch.arange(x.shape[0], device=x.device)
    return order, score


def main():
    p = make_common_parser("Original data baseline")
    args = p.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)
    x = generate_random_qlc(
        args.n_pages, args.page_cells, args.seed, device
    )
    lut = build_score_lut(device, alpha=args.alpha)

    _, score = original_baseline(x, lut, args.score_chunk)
    s = float(score)
    print_result("Original data", s, s)


if __name__ == "__main__":
    main()
