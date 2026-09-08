#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DVDS-style endurance-dependent distribution-shaping baseline.

IMPORTANT REPRODUCTION NOTE
---------------------------
The PDA-LSTM paper compares DVDS at CL=64 and CL=128 but does not include
the original DVDS state mapping table or endurance-dependent equations.
The cited DVDS work is for 3D-TLC, while PDA-LSTM uses QLC (16 states).

This is therefore a RUNNABLE QLC PROXY, not bit-exact DVDS.

Proxy design:
- split each page into CL-cell chunks,
- construct several invertible QLC state permutations,
- compute the chunk histogram with torch.bincount/one_hot-style reduction,
- choose the mapping minimizing an endurance-dependent state penalty,
- store the selected mapping index as metadata.

The penalty increasingly discourages high Vth states as PE cycles increase.
"""

import torch
import torch.nn.functional as F
from common_lcm import (
    make_common_parser, resolve_device, set_seed, generate_random_qlc,
    build_score_lut, score_order_direct, print_result
)


def dvds_candidate_maps(device):
    """
    Four invertible 16-state mappings.
    Rows: mapping id; columns: original state; value: programmed state.
    """
    base = torch.arange(16, device=device, dtype=torch.long)

    maps = torch.stack([
        base,                         # identity
        15 - base,                    # mirror
        (base + 4) % 16,              # cyclic shape 1
        (base + 8) % 16,              # cyclic shape 2
    ], dim=0)
    return maps


@torch.no_grad()
def dvds_proxy(
    x: torch.Tensor,
    code_length: int = 64,
    pe_cycles: int = 500,
):
    if code_length <= 0:
        raise ValueError("code_length must be positive")

    n, d = x.shape
    pad = (-d) % code_length

    if pad:
        xp = F.pad(x, (0, pad), mode="constant", value=0)
    else:
        xp = x

    nc = xp.shape[1] // code_length
    blocks = xp.view(n, nc, code_length).long()  # [N,C,L]

    # Histograms [N,C,16].
    hist = F.one_hot(blocks, num_classes=16).sum(dim=-2).float()

    maps = dvds_candidate_maps(x.device)  # [M,16]

    # Endurance-dependent penalty:
    # base term penalizes programmed level;
    # endurance term increasingly penalizes the upper tail.
    state = torch.arange(16, device=x.device, dtype=torch.float32)
    endurance = max(float(pe_cycles), 0.0) / 1000.0
    penalty = state + endurance * (state / 15.0) ** 2 * 15.0

    # Cost of applying each mapping:
    # mapped_penalty[m, original_state] = penalty[maps[m, state]]
    mapped_penalty = penalty[maps]  # [M,16]

    # Einstein contraction:
    # hist[n,c,s] x cost[m,s] -> total[n,c,m]
    costs = torch.einsum("ncs,ms->ncm", hist, mapped_penalty)
    choice = torch.argmin(costs, dim=-1)  # [N,C]

    # Apply chosen mapping without Python loops over chunks.
    selected_maps = maps[choice]  # [N,C,16]
    ni = torch.arange(n, device=x.device)[:, None, None]
    ci = torch.arange(nc, device=x.device)[None, :, None]
    out = selected_maps[ni, ci, blocks]  # [N,C,L]

    out = out.reshape(n, -1)[:, :d].to(torch.uint8)
    return out, choice


def main():
    p = make_common_parser("DVDS-style QLC proxy baseline")
    p.add_argument("--cl", type=int, choices=[64, 128], default=64)
    p.add_argument("--pe-cycles", type=int, default=500)
    args = p.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)

    x = generate_random_qlc(args.n_pages, args.page_cells, args.seed, device)
    lut = build_score_lut(device, alpha=args.alpha)

    original = score_order_direct(x, score_lut=lut, chunk_size=args.score_chunk)
    y, choice = dvds_proxy(
        x, code_length=args.cl, pe_cycles=args.pe_cycles
    )
    score = score_order_direct(y, score_lut=lut, chunk_size=args.score_chunk)

    print_result(f"DVDS-style proxy CL{args.cl}", float(original), float(score))
    print("mapping units   :", int(choice.numel()))
    print("mapping counts  :", torch.bincount(
        choice.flatten(), minlength=4
    ).detach().cpu().tolist())


if __name__ == "__main__":
    main()
