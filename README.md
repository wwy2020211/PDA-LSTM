# PDA-LSTM comparison baseline code

This directory contains one runnable Python implementation for each comparison
family used/discussed in the PDA-LSTM paper, with random QLC input.

## Files

- `common_lcm.py`  
  Random QLC data generator and the paper's LCM score, including the 16×16×16
  score LUT, direct score reduction, and the N×N×N page-triple score cube.

- `baseline_original.py`  
  Original data / no processing.

- `baseline_random_bitline.py`  
  Random movement in the bitline direction. Each page independently permutes
  its cell positions, changing vertical neighboring-cell patterns.

- `baseline_random_wordline.py`  
  Random wordline/page sorting. Thousands of random permutations can be scored
  in parallel from a precomputed N³ score cube. No N! enumeration.

- `baseline_greedy.py`  
  Greedy wordline sequence generation. Because PDA's objective is based on
  three adjacent pages, the next page maximizes the score with the previous two.
  All ordered starting pairs are tried.

- `baseline_tsp_sa.py`  
  TSP-style simulated annealing using parallel independent chains on GPU.
  The paper discusses SA as a TSP solver but does not state the exact solver
  configuration used in its experiment, so this is an explicit reproduction
  choice rather than a claim of hidden author hyperparameters.

- `baseline_wbvm.py`  
  WBVM-style QLC proxy with CL64/CL128.

- `baseline_dvds.py`  
  DVDS-style QLC proxy with CL64/CL128 and a PE-cycle parameter.

- `run_all_baselines.py`  
  Runs all baselines on exactly the same random sample and prints a comparison
  table.

## What is directly supported by the PDA-LSTM paper?

The paper explicitly states that it compares:
- WBVM and DVDS at different code lengths,
- random sorting in bitline and wordline directions,
- Greedy,
- TSP,
- Original/unprocessed data.

It also says code length is the number of cells handled by a WBVM/DVDS unit.

The paper's background describes:
- Greedy as nearest-neighbor sequence construction.
- TSP and simulated annealing as a sequence-search approach.
- TSP/Greedy repeatedly calculate LCM score during search.

## Important limitation: WBVM and DVDS

The PDA-LSTM PDF does **not** reproduce the exact WBVM or DVDS mapping tables,
BER-score tables, endurance equations, or their TLC-to-QLC conversion.
The cited original works are TLC methods, whereas PDA-LSTM's model/data use QLC
states 0..15.

Therefore the included WBVM/DVDS files are intentionally labeled **QLC proxy**
implementations. They are useful for:
- testing the code-length mechanism,
- reproducing the comparison pipeline,
- measuring score/overhead trends,
- plugging in the exact mapping tables later.

They should **not** be presented as bit-exact implementations of the cited
WBVM/DVDS papers.

The WBVM proxy uses the externally reported structural idea of chunk-level
flipping with a flag bit and state-count/Vth-based decision.

## Requirements

```bash
pip install torch
```

CUDA is strongly recommended for full paper-size data.

## Paper-size random input

The default is:

```text
N = 16 pages
D = 18 * 1024 * 8 = 147456 cells/page
state = integer 0..15
```

Run all:

```bash
python run_all_baselines.py --device cuda
```

A faster smoke test:

```bash
python run_all_baselines.py \
  --page-cells 2048 \
  --random-wordline-trials 512 \
  --random-bitline-trials 2 \
  --sa-iterations 300 \
  --sa-chains 64
```

Individual examples:

```bash
python baseline_original.py --device cuda
python baseline_random_wordline.py --device cuda --trials 4096
python baseline_random_bitline.py --device cuda --trials 8
python baseline_greedy.py --device cuda
python baseline_tsp_sa.py --device cuda --iterations 5000 --chains 256
python baseline_wbvm.py --device cuda --cl 64
python baseline_wbvm.py --device cuda --cl 128
python baseline_dvds.py --device cuda --cl 64 --pe-cycles 500
python baseline_dvds.py --device cuda --cl 128 --pe-cycles 500
```

## Matrix / parallel-computing choices

The wordline algorithms do not enumerate all permutations.

`common_lcm.py` first computes:

```text
S_AC[N,N,N]
```

only 4096 ordered triples for N=16, using chunked broadcasting and LUT lookup.

Thousands of random full orders are then scored together using tensor advanced
indexing. The SA implementation evolves many independent permutation chains in
parallel.

DVDS's histogram-to-candidate-map cost uses Einstein summation:

```python
costs = torch.einsum("ncs,ms->ncm", hist, mapped_penalty)
```

This keeps the implementation consistent with the tensor-first approach used
for the PDA-LSTM reproduction.
