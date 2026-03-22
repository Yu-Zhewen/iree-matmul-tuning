# IREE Matmul Tuning Analysis

## Problem Setup

- **Shapes**: 512x512, 1024x1024, 2048x2048, 4096x4096, 8192x8192, 16384x16384 square BF16 matmuls
- **Layout**: LHS (MxK), RHS transposed (NxK), Output (MxN)
- **Data types**: BF16 inputs, F32 accumulation
- **GPU**: AMD MI355X (gfx950), 256 CUs
- **Compiler**: IREE ([`benchmark_multi_buffer`](https://github.com/Yu-Zhewen/iree/tree/benchmark_multi_buffer) branch) with `--iree-llvmgpu-use-direct-load`
- **Tuner**: `amdsharktuner` with 5000 candidates per shape

## Repository Structure

```
.
├── README.md
├── mlir/
│   ├── 512/
│   │   ├── template.mlir        # Executable-sources MLIR (transposed RHS)
│   │   ├── 1939_spec.mlir       # Tuning specs for top 10
│   │   └── ...
│   ├── 1024/ ... 2048/ ... 4096/ ... 8192/ ... 16384/
├── vmfb/
│   ├── 512/
│   │   ├── 512_heur_gfx950.vmfb
│   │   ├── 512_heur_mi355x.vmfb
│   │   ├── 512_cid_1939.vmfb    # Top 10 tuned VMFBs
│   │   └── ...
│   ├── 1024/ ... 2048/ ... 4096/ ... 8192/ ... 16384/
├── traces/                      # ATT thread traces (top 3 + heuristics)
│   ├── 512/ ... 1024/ ... 2048/ ... 4096/ ... 8192/ ... 16384/
└── profiles/                    # rocprof-compute stats (top 3 + heuristics)
    ├── 512/ ... 1024/ ... 2048/ ... 4096/ ... 8192/ ... 16384/
```

## Configurations

Top 10 candidates per case (out of 5000 candidates per run, `--num-candidates=5000`). All runs use `--iree-llvmgpu-use-direct-load` and fixed pipeline options: `prefetch_num_stages = 2`, `no_reduce_shared_memory_bank_conflicts = true`, `amdgpu-waves-per-eu = 2`.

### Knob Definitions

| Knob | Meaning |
| --- | --- |
| tile\_m, tile\_n | Workgroup tile sizes along M and N |
| tile\_k | Reduction tile size along K |
| sg\_m, sg\_n | Subgroup count along M and N |
| intrinsic | MFMA instruction variant (MxNxK) |

### Derived Metrics

| Metric | Definition |
| --- | --- |
| time (us) | Time reported by the tuner |
| time reproduced (us) | Re-benchmarked time using `iree-benchmark-module` |
| numWG | Number of workgroups: `ceil(M/tile_m) * ceil(N/tile_n)` |
| waves | Number of dispatch waves: `ceil(numWG / wgpCount)` |
| util | Compute utilization: `numWG / (waves * wgpCount)` |

`wgpCount = 256` is the number of Compute Units on MI355X (32 CUs per XCD × 8 XCDs).

### MediumGemm: 512x512 (heuristic-gfx950 = 61.0 us)

| Rank | cid | tile\_m | tile\_n | tile\_k | sg\_m | sg\_n | intrinsic | time (us) | vs heuristic-gfx950 | time reproduced (us) | numWG | waves | util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **heuristic-gfx950** | **--** | **64** | **128** | **4** | **2** | **2** | **16x16x32** | **61.0** | **100%** | **50** | **32** | **1** | **12.5%** |
| heuristic-mi355x | -- | 32 | 32 | 4 | 2 | 2 | 16x16x32 | 56.0 | 92% | 51 | 256 | 1 | 100% |
| 1 | 1939 | 32 | 64 | 1 | 1 | 2 | 32x32x16 | 41.33 | **67.8%** | 53 | 128 | 1 | 50% |
| 2 | 248 | 64 | 64 | 16 | 1 | 1 | 16x16x32 | 47.33 | 77.6% | 57 | 64 | 1 | 25% |
| 3 | 2121 | 128 | 64 | 4 | 1 | 2 | 16x16x32 | 48.33 | 79.2% | 60 | 32 | 1 | 12.5% |
| 4 | 1864 | 16 | 128 | 1 | 1 | 2 | 16x16x32 | 48.67 | 79.8% | 52 | 128 | 1 | 50% |
| 5 | 2280 | 64 | 64 | 1 | 1 | 2 | 16x16x32 | 48.67 | 79.8% | 60 | 64 | 1 | 25% |
| 6 | 1554 | 128 | 32 | 2 | 1 | 1 | 16x16x32 | 49.00 | 80.3% | 56 | 64 | 1 | 25% |
| 7 | 2366 | 32 | 16 | 1 | 2 | 1 | 16x16x16 | 49.00 | 80.3% | 57 | 512 | 2 | 100% |
| 8 | 3201 | 16 | 256 | 4 | 1 | 4 | 16x16x32 | 49.33 | 80.9% | 53 | 64 | 1 | 25% |
| 9 | 117 | 256 | 64 | 1 | 1 | 2 | 16x16x32 | 49.67 | 81.4% | 59 | 16 | 1 | 6.2% |
| 10 | 290 | 32 | 64 | 1 | 2 | 4 | 16x16x32 | 49.67 | 81.4% | 57 | 128 | 1 | 50% |

### MediumGemm: 1024x1024 (heuristic-gfx950 = 65.0 us)

| Rank | cid | tile\_m | tile\_n | tile\_k | sg\_m | sg\_n | intrinsic | time (us) | vs heuristic-gfx950 | time reproduced (us) | numWG | waves | util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **heuristic-gfx950** | **--** | **64** | **128** | **4** | **2** | **2** | **16x16x32** | **65.0** | **100%** | **75** | **128** | **1** | **50%** |
| heuristic-mi355x | -- | 64 | 64 | 4 | 2 | 2 | 16x16x32 | 51.0 | 78% | 54 | 256 | 1 | 100% |
| 1 | 2535 | 128 | 32 | 2 | 4 | 1 | 32x32x16 | 49.67 | **76.4%** | 55 | 256 | 1 | 100% |
| 2 | 2123 | 64 | 32 | 2 | 1 | 1 | 32x32x16 | 50.33 | 77.4% | 55 | 512 | 2 | 100% |
| 3 | 1767 | 128 | 16 | 4 | 2 | 1 | 16x16x32 | 50.67 | 77.9% | 61 | 512 | 2 | 100% |
| 4 | 424 | 32 | 32 | 1 | 2 | 1 | 16x16x32 | 51.00 | 78.5% | 58 | 1024 | 4 | 100% |
| 5 | 1718 | 128 | 32 | 2 | 1 | 2 | 16x16x32 | 51.33 | 78.9% | 62 | 256 | 1 | 100% |
| 6 | 2376 | 64 | 32 | 1 | 1 | 2 | 16x16x32 | 51.33 | 78.9% | 60 | 512 | 2 | 100% |
| 7 | 84 | 32 | 64 | 1 | 1 | 1 | 32x32x16 | 51.67 | 79.5% | 70 | 512 | 2 | 100% |
| 8 | 1690 | 32 | 64 | 2 | 1 | 2 | 32x32x16 | 51.67 | 79.5% | 56 | 512 | 2 | 100% |
| 9 | 1662 | 128 | 16 | 2 | 2 | 1 | 16x16x32 | 52.00 | 80.0% | 60 | 512 | 2 | 100% |
| 10 | 759 | 32 | 64 | 2 | 1 | 2 | 16x16x32 | 52.33 | 80.5% | 54 | 512 | 2 | 100% |

### MediumGemm(mi355x)/LargeGemm(gfx950): 2048x2048 (heuristic-gfx950 = 86.0 us)

| Rank | cid | tile\_m | tile\_n | tile\_k | sg\_m | sg\_n | intrinsic | time (us) | vs heuristic-gfx950 | time reproduced (us) | numWG | waves | util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **heuristic-gfx950** | **--** | **128** | **128** | **1** | **2** | **2** | **16x16x32** | **86.0** | **100%** | **86** | **256** | **1** | **100%** |
| heuristic-mi355x | -- | 64 | 128 | 4 | 2 | 2 | 16x16x32 | 125.0 | 145% | 117 | 512 | 2 | 100% |
| 1 | 2987 | 64 | 64 | 2 | 1 | 1 | 16x16x32 | 73.33 | **85.3%** | 73 | 1024 | 4 | 100% |
| 2 | 2398 | 64 | 128 | 1 | 4 | 1 | 16x16x32 | 75.67 | 88.0% | 81 | 512 | 2 | 100% |
| 3 | 2767 | 128 | 128 | 2 | 2 | 4 | 16x16x32 | 78.67 | 91.5% | 83 | 256 | 1 | 100% |
| 4 | 1795 | 128 | 128 | 1 | 4 | 2 | 16x16x32 | 79.00 | 91.9% | 88 | 256 | 1 | 100% |
| 5 | 1578 | 32 | 128 | 2 | 1 | 1 | 16x16x32 | 79.33 | 92.2% | 78 | 1024 | 4 | 100% |
| 6 | 3496 | 32 | 128 | 2 | 1 | 1 | 32x32x16 | 79.33 | 92.2% | 93 | 1024 | 4 | 100% |
| 7 | 219 | 256 | 32 | 2 | 2 | 1 | 16x16x32 | 80.00 | 93.0% | 90 | 512 | 2 | 100% |
| 8 | 1182 | 64 | 128 | 1 | 2 | 1 | 16x16x32 | 81.33 | 94.6% | 90 | 512 | 2 | 100% |
| 9 | 1710 | 128 | 64 | 2 | 1 | 2 | 16x16x32 | 81.33 | 94.6% | 90 | 512 | 2 | 100% |
| 10 | 768 | 256 | 32 | 2 | 4 | 1 | 16x16x32 | 82.00 | 95.3% | 91 | 512 | 2 | 100% |

### LargeGemm: 4096x4096 (heuristic-gfx950 = 201.0 us)

| Rank | cid | tile\_m | tile\_n | tile\_k | sg\_m | sg\_n | intrinsic | time (us) | vs heuristic-gfx950 | time reproduced (us) | numWG | waves | util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **heuristic-gfx950** | **--** | **128** | **128** | **1** | **2** | **2** | **16x16x32** | **201.0** | **100%** | **204** | **1024** | **4** | **100%** |
| heuristic-mi355x | -- | 128 | 128 | 1 | 2 | 2 | 16x16x32 | 207.0 | 103% | 204 | 1024 | 4 | 100% |
| heuristic-mi355x-pr#23652 | -- | 128 | 256 | 1 | 2 | 2 | 16x16x32 | 196.0 | 97% | | 512 | 2 | 100% |
| 1 | 577 | 128 | 256 | 2 | 1 | 4 | 32x32x16 | 160.67 | **79.9%** | 174 | 512 | 2 | 100% |
| 2 | 1810 | 256 | 256 | 1 | 4 | 2 | 16x16x32 | 168.00 | 83.6% | 178 | 256 | 1 | 100% |
| 3 | 1468 | 256 | 128 | 2 | 2 | 2 | 32x32x16 | 168.33 | 83.7% | 182 | 512 | 2 | 100% |
| 4 | 1606 | 256 | 128 | 1 | 1 | 4 | 16x16x32 | 169.67 | 84.4% | 176 | 512 | 2 | 100% |
| 5 | 2563 | 128 | 256 | 1 | 4 | 1 | 16x16x32 | 170.00 | 84.6% | 195 | 512 | 2 | 100% |
| 6 | 2362 | 128 | 256 | 2 | 2 | 2 | 32x32x16 | 170.67 | 84.9% | 175 | 512 | 2 | 100% |
| 7 | 2264 | 256 | 256 | 1 | 2 | 4 | 16x16x32 | 177.67 | 88.4% | 178 | 256 | 1 | 100% |
| 8 | 410 | 128 | 256 | 1 | 1 | 8 | 16x16x32 | 178.00 | 88.6% | 189 | 512 | 2 | 100% |
| 9 | 514 | 128 | 128 | 2 | 2 | 1 | 16x16x32 | 178.67 | 88.9% | 190 | 1024 | 4 | 100% |
| 10 | 1355 | 256 | 128 | 2 | 4 | 1 | 32x32x16 | 178.67 | 88.9% | 184 | 512 | 2 | 100% |

### LargeGemm: 8192x8192 (heuristic-gfx950 = 1580.0 us)

| Rank | cid | tile\_m | tile\_n | tile\_k | sg\_m | sg\_n | intrinsic | time (us) | vs heuristic-gfx950 | time reproduced (us) | numWG | waves | util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **heuristic-gfx950** | **--** | **128** | **128** | **1** | **2** | **2** | **16x16x32** | **1580.0** | **100%** | **1600** | **4096** | **16** | **100%** |
| heuristic-mi355x | -- | 128 | 128 | 1 | 2 | 2 | 16x16x32 | 1520.0 | 96% | 1560 | 4096 | 16 | 100% |
| heuristic-mi355x-pr#23652 | -- | 128 | 256 | 1 | 2 | 2 | 16x16x32 | 1170.0 | 74% | | 2048 | 8 | 100% |
| 1 | 3392 | 256 | 256 | 1 | 4 | 2 | 16x16x32 | 975.00 | **61.7%** | 967 | 1024 | 4 | 100% |
| 2 | 170 | 128 | 256 | 1 | 1 | 8 | 16x16x32 | 980.33 | 62.0% | 1000 | 2048 | 8 | 100% |
| 3 | 1243 | 256 | 256 | 1 | 2 | 4 | 16x16x32 | 991.33 | 62.7% | 987 | 1024 | 4 | 100% |
| 4 | 902 | 128 | 128 | 2 | 2 | 1 | 16x16x32 | 1005.00 | 63.6% | 1000 | 4096 | 16 | 100% |
| 5 | 15 | 256 | 128 | 1 | 2 | 4 | 16x16x32 | 1013.33 | 64.1% | 1010 | 2048 | 8 | 100% |
| 6 | 1369 | 128 | 128 | 2 | 1 | 2 | 16x16x32 | 1016.67 | 64.3% | 1010 | 4096 | 16 | 100% |
| 7 | 2793 | 128 | 256 | 2 | 1 | 4 | 32x32x16 | 1016.67 | 64.3% | 1010 | 2048 | 8 | 100% |
| 8 | 1891 | 256 | 128 | 1 | 4 | 2 | 16x16x32 | 1033.33 | 65.4% | 1030 | 2048 | 8 | 100% |
| 9 | 79 | 256 | 256 | 1 | 1 | 8 | 16x16x32 | 1036.67 | 65.6% | 1040 | 1024 | 4 | 100% |
| 10 | 2300 | 128 | 256 | 2 | 2 | 2 | 32x32x16 | 1040.00 | 65.8% | 1060 | 2048 | 8 | 100% |

### VeryLargeGemm: 16384x16384 (heuristic-gfx950 = 9910.0 us)

| Rank | cid | tile\_m | tile\_n | tile\_k | sg\_m | sg\_n | intrinsic | time (us) | vs heuristic-gfx950 | time reproduced (us) | numWG | waves | util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **heuristic-gfx950** | **--** | **256** | **256** | **2** | **2** | **2** | **32x32x16** | **9910.0** | **100%** | **9960** | **4096** | **16** | **100%** |
| heuristic-mi355x | -- | 256 | 256 | 2 | 2 | 2 | 32x32x16 | 9930.0 | 100% | 9910 | 4096 | 16 | 100% |
| heuristic-mi355x-pr#23652 | -- | 256 | 512 | 2 | 2 | 2 | 32x32x16 | 104000.0 | 1049% | | 2048 | 8 | 100% |
| 1 | 1574 | 256 | 128 | 1 | 2 | 2 | 16x16x32 | 8013.33 | **80.9%** | 8020 | 8192 | 32 | 100% |
| 2 | 794 | 256 | 128 | 1 | 4 | 1 | 16x16x32 | 8063.33 | 81.4% | 8180 | 8192 | 32 | 100% |
| 3 | 418 | 128 | 128 | 2 | 1 | 2 | 16x16x32 | 8156.67 | 82.3% | 8230 | 16384 | 64 | 100% |
| 4 | 737 | 256 | 128 | 1 | 1 | 4 | 16x16x32 | 8173.33 | 82.5% | 8260 | 8192 | 32 | 100% |
| 5 | 2530 | 256 | 128 | 2 | 4 | 1 | 32x32x16 | 8373.33 | 84.5% | 8330 | 8192 | 32 | 100% |
| 6 | 3049 | 256 | 128 | 2 | 2 | 2 | 32x32x16 | 8430.00 | 85.1% | 8400 | 8192 | 32 | 100% |
| 7 | 162 | 128 | 128 | 2 | 2 | 1 | 16x16x32 | 8513.33 | 85.9% | 8180 | 16384 | 64 | 100% |
| 8 | 3160 | 256 | 256 | 1 | 4 | 2 | 16x16x32 | 8793.33 | 88.7% | 8810 | 4096 | 16 | 100% |
| 9 | 43 | 256 | 128 | 1 | 4 | 2 | 16x16x32 | 8816.67 | 89.0% | 8760 | 8192 | 32 | 100% |
| 10 | 578 | 256 | 256 | 1 | 2 | 4 | 16x16x32 | 8853.33 | 89.3% | 8860 | 4096 | 16 | 100% |

## Reproducing

### Compile a tuned config

```bash
iree-compile mlir/<size>/template.mlir \
  --compile-from=executable-sources \
  --iree-codegen-tuning-spec-path=mlir/<size>/<cid>_spec.mlir \
  --iree-hal-target-device=hip \
  --iree-rocm-target=mi355x \
  --iree-llvmgpu-use-direct-load \
  --iree-codegen-enable-default-tuning-specs=false \
  -o output.vmfb
```

### Compile a heuristic config

```bash
iree-compile mlir/<size>/template.mlir \
  --compile-from=executable-sources \
  --iree-hal-target-device=hip \
  --iree-rocm-target=gfx950 \
  --iree-llvmgpu-use-direct-load \
  --iree-codegen-enable-default-tuning-specs=false \
  -o heuristic.vmfb
```

### Benchmark

```bash
iree-benchmark-module --device=hip --module=output.vmfb \
  --function=mm_<size> --input=<size>x<size>xbf16 --input=<size>x<size>xbf16 \
  --benchmark_repetitions=5
```

### Generate ATT thread traces

```bash
rocprofv3 --att \
  --att-shader-engine-mask 0xF \
  --att-consecutive-kernels 10 \
  -d trace_output/ \
  -- iree-benchmark-module --device=hip --module=output.vmfb \
  --function=mm_<size> --input=<size>x<size>xbf16 --input=<size>x<size>xbf16 \
  --benchmark_repetitions=3
```

### Generate rocprof-compute stats

```bash
rocprof-compute profile -n 3 \
  -o profile_output/ \
  -- iree-benchmark-module --device=hip --module=output.vmfb \
  --function=mm_<size> --input=<size>x<size>xbf16 --input=<size>x<size>xbf16 \
  --benchmark_repetitions=3

rocprof-compute analyze -p profile_output/ > stats.txt
```
