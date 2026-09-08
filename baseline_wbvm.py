#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WBVM-style code-length baseline for QLC random data.

IMPORTANT REPRODUCTION NOTE
---------------------------
The PDA-LSTM paper compares WBVM at CL=64 and CL=128, and says:
- WBVM is an intra-page mapping method,
- code length (CL) is the number of cells handled by one code unit,
- extra flag bits record the mapping.

However, the PDA-LSTM PDF does NOT provide WBVM's exact TLC->QLC mapping table,
BER-score table, or its exact implementation. The original WBVM papers are TLC.

Therefore this script is a RUNNABLE QLC PROXY, not a claim of bit-exact WBVM:
- split each page into CL-cell chunks,
- one binary flag per chunk,
- choose identity vs complemented QLC-state mapping (x -> 15-x),
- select the mapping with the lower programmed-level/Vth proxy cost.

This preserves the key "chunk + flip flag" structure reported for WBVM and
allows CL64/CL128 experiments on the same random QLC input.
"""

import torch
from common_lcm import (
    make_common_parser, resolve_device, set_seed, generate_random_qlc,
    build_score_lut, score_order_direct, print_result
)


@torch.no_grad()
def wbvm_proxy(x: torch.Tensor, code_length: int = 64):
    """
    Binary chunk modulation:
      candidate 0: x
      candidate 1: 15-x

    Vth proxy cost = sum(program level).
    Pick lower-cost candidate independently for each [page, chunk].

    Returns:
      y [N,D]
      flags [N,n_chunks] (0 identity, 1 complement)
    """
    if code_length <= 0:
        raise ValueError("code_length must be positive")

    n, d = x.shape
    pad = (-d) % code_length

    if pad:
        xp = torch.nn.functional.pad(
            x, (0, pad), mode="constant", value=0
        )
    else:
        xp = x

    nc = xp.shape[1] // code_length
    blocks = xp.view(n, nc, code_length).long()

    identity_cost = blocks.sum(dim=-1)
    flipped = 15 - blocks
    flipped_cost = flipped.sum(dim=-1)

    flags = (flipped_cost < identity_cost)
    out = torch.where(flags[..., None], flipped, blocks)
    out = out.reshape(n, -1)[:, :d].to(torch.uint8)

    return out, flags


def main():
    p = make_common_parser("WBVM-style QLC proxy baseline")
    p.add_argument("--cl", type=int, choices=[64, 128], default=64)
    args = p.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)

    x = generate_random_qlc(args.n_pages, args.page_cells, args.seed, device)
    lut = build_score_lut(device, alpha=args.alpha)

    original = score_order_direct(x, score_lut=lut, chunk_size=args.score_chunk)
    y, flags = wbvm_proxy(x, code_length=args.cl)
    score = score_order_direct(y, score_lut=lut, chunk_size=args.score_chunk)

    print_result(f"WBVM-style proxy CL{args.cl}", float(original), float(score))
    print("flag bits       :", int(flags.numel()))
    print("flipped chunks  :", int(flags.sum()))


if __name__ == "__main__":
    main()
