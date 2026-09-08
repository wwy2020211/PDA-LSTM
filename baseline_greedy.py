#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy wordline arrangement baseline.

The paper describes Greedy as nearest-neighbor selection:
at each step select the best unvisited neighbor.

Because PDA's physical objective is a THREE-page LCM score, the natural
paper-consistent greedy extension uses the previous two pages and selects
the candidate k maximizing S_AC[prev2, prev1, k].

To avoid an arbitrary starting city/page, all ordered starting pairs are tried.
This is O(N^4)-scale greedy work for N=16, not N! enumeration.
"""

import torch
from common_lcm import (
    make_common_parser, resolve_device, set_seed, generate_random_qlc,
    build_score_lut, build_page_triple_score_cube,
    score_order_direct, score_orders_from_cube, print_result
)


@torch.no_grad()
def greedy_from_start_pair(s_ac, first, second):
    n = s_ac.shape[0]
    order = [int(first), int(second)]
    used = torch.zeros(n, dtype=torch.bool, device=s_ac.device)
    used[first] = True
    used[second] = True

    while len(order) < n:
        candidates = torch.where(~used)[0]
        prev2, prev1 = order[-2], order[-1]
        local_scores = s_ac[prev2, prev1, candidates]
        best = candidates[torch.argmax(local_scores)]
        b = int(best)
        order.append(b)
        used[b] = True

    return torch.tensor(order, device=s_ac.device, dtype=torch.long)


@torch.no_grad()
def greedy_search(s_ac):
    n = s_ac.shape[0]
    candidate_orders = []

    for i in range(n):
        for j in range(n):
            if i != j:
                candidate_orders.append(greedy_from_start_pair(s_ac, i, j))

    orders = torch.stack(candidate_orders, dim=0)
    scores = score_orders_from_cube(s_ac, orders)
    idx = torch.argmax(scores)
    return orders[idx], scores[idx]


def main():
    p = make_common_parser("Greedy wordline baseline")
    args = p.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)

    x = generate_random_qlc(args.n_pages, args.page_cells, args.seed, device)
    lut = build_score_lut(device, alpha=args.alpha)

    original = score_order_direct(x, score_lut=lut, chunk_size=args.score_chunk)
    cube = build_page_triple_score_cube(
        x, score_lut=lut, chunk_size=args.score_chunk
    )

    order, score = greedy_search(cube)

    print_result("Greedy", float(original), float(score))
    print("best order      :", order.detach().cpu().tolist())


if __name__ == "__main__":
    main()
