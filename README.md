# IREE Matmul Tuning Analysis

## Problem Setup

- **Shapes**: 512x512 and 1024x1024 square BF16 matmuls
- **Layout**: LHS (MxK), RHS transposed (NxK), Output (MxN)
- **Data types**: BF16 inputs, F32 accumulation
- **GPU**: AMD MI355X (gfx950), 256 WGPs
- **Compiler**: IREE with `--iree-llvmgpu-use-direct-load`
- **Tuner**: `amdsharktuner` with 5000 candidates per shape

## Repository Structure

```
.
├── README.md
├── mlir/
│   ├── 512/
│   │   ├── template.mlir        # Executable-sources MLIR for 512x512 (transposed RHS)
│   │   ├── 1058_spec.mlir       # Tuning specs
│   │   └── ...
│   └── 1024/
│       ├── template.mlir        # Executable-sources MLIR for 1024x1024 (transposed RHS)
│       ├── 1591_spec.mlir       # Tuning specs
│       └── ...
├── vmfb/
│   ├── 512/                     # Compiled VMFBs for 512x512
│   │   ├── repro_512_1058.vmfb
│   │   ├── repro_heur_gfx950_512.vmfb
│   │   └── ...
│   └── 1024/                    # Compiled VMFBs for 1024x1024
│       └── ...
└── traces/
    ├── 512/                     # ATT thread traces for 512x512
    │   ├── 512_cid_1058/
    │   │   ├── stats_*.csv      # Per-instruction hit/latency/stall stats
    │   │   └── ui_output_*/     # Decoded wave traces (code.json, wstates, etc.)
    │   └── ...
    └── 1024/                    # ATT thread traces for 1024x1024
        └── ...
```

## Configurations

Top 10 candidates per case (out of 5000 candidates per run, `--num-candidates=5000`). All runs use `--iree-llvmgpu-use-direct-load` and fixed pipeline options: `prefetch_num_stages = 2`, `no_reduce_shared_memory_bank_conflicts = true`, `amdgpu-waves-per-eu = 2`.

### Knob Definitions

| Knob | Meaning |
| --- | --- |
| tile_m, tile_n | Workgroup tile sizes along M and N |
| tile_k | Reduction tile size along K |
| sg_m, sg_n | Subgroup count along M and N |
| intrinsic | MFMA instruction variant (MxNxK) |

### Derived Metrics

| Metric | Definition |
| --- | --- |
| time (us) | Original time reported by the tuner |
| time reproduced (us) | Re-benchmarked time using the tuner's template + spec with the latest IREE build |
| numWG | Number of workgroups: `ceil(M/tile_m) * ceil(N/tile_n)` |
| waves | Number of dispatch waves: `ceil(numWG / wgpCount)` |
| util | Compute utilization: `numWG / (waves * wgpCount)` |

`wgpCount = 256` is the number of Compute Units on MI355X (32 CUs per XCD × 8 XCDs).

### MediumGemm: 512x512 (heuristic-gfx950 = 45.8 us)

| Rank | cid | tile_m | tile_n | tile_k | sg_m | sg_n | intrinsic | time (us) | vs heuristic-gfx950 | time reproduced (us) | numWG | waves | util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **heuristic-gfx950** | **--** | **64** | **128** | **4** | **2** | **2** | **16x16x32** | **45.8** | **100%** | **44.6** | **32** | **1** | **12.5%** |
| heuristic-mi355x | -- | 32 | 32 | 4 | 2 | 2 | 16x16x32 | | 89% | 40.6 | 256 | 1 | 100% |
| 1 | 1058 | 16 | 32 | 4 | 1 | 2 | 16x16x32 | 28.33 | **61.8%** | 39.7 | 512 | 2 | 100% |
| 2 | 1101 | 16 | 128 | 2 | 1 | 2 | 16x16x32 | 30.33 | 66.2% | 32.3 | 128 | 1 | 50% |
| 3 | 1253 | 128 | 128 | 1 | 4 | 2 | 32x32x16 | 31.33 | 68.4% | 39.0 | 16 | 1 | 6.2% |
| 4 | 981 | 64 | 32 | 16 | 1 | 2 | 16x16x32 | 32.00 | 69.8% | 35.6 | 128 | 1 | 50% |
| 5 | 998 | 32 | 64 | 1 | 2 | 1 | 16x16x32 | 32.00 | 69.8% | 38.6 | 128 | 1 | 50% |
| 6 | 1004 | 64 | 16 | 1 | 2 | 1 | 16x16x16 | 32.00 | 69.8% | 39.7 | 256 | 1 | 100% |
| 7 | 1047 | 32 | 64 | 4 | 1 | 1 | 16x16x32 | 32.00 | 69.8% | 45.0 | 128 | 1 | 50% |
| 8 | 1049 | 256 | 16 | 8 | 8 | 1 | 16x16x32 | 32.00 | 69.8% | 45.4 | 64 | 1 | 25% |
| 9 | 897 | 32 | 64 | 1 | 1 | 1 | 16x16x32 | 32.33 | 70.5% | 40.5 | 128 | 1 | 50% |
| 10 | 982 | 32 | 256 | 1 | 1 | 8 | 16x16x32 | 32.33 | 70.5% | 39.5 | 32 | 1 | 12.5% |

### MediumGemm: 1024x1024 (heuristic-gfx950 = 54.7 us)

| Rank | cid | tile_m | tile_n | tile_k | sg_m | sg_n | intrinsic | time (us) | vs heuristic-gfx950 | time reproduced (us) | numWG | waves | util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **heuristic-gfx950** | **--** | **64** | **128** | **4** | **2** | **2** | **16x16x32** | **54.7** | **100%** | **49.3** | **128** | **1** | **50%** |
| heuristic-mi355x | -- | 64 | 64 | 4 | 2 | 2 | 16x16x32 | | 97% | 51.6 | 256 | 1 | 100% |
| 1 | 1591 | 64 | 32 | 1 | 1 | 1 | 16x16x32 | 34.00 | **62.2%** | 47.1 | 512 | 2 | 100% |
| 2 | 860 | 64 | 64 | 1 | 2 | 1 | 16x16x32 | 37.00 | 67.7% | 41.1 | 256 | 1 | 100% |
| 3 | 3348 | 32 | 32 | 2 | 2 | 1 | 16x16x32 | 37.33 | 68.3% | 46.1 | 1024 | 4 | 100% |
| 4 | 85 | 64 | 64 | 2 | 2 | 2 | 16x16x32 | 39.33 | 71.9% | 39.5 | 256 | 1 | 100% |
| 5 | 27 | 32 | 128 | 1 | 2 | 4 | 16x16x32 | 39.67 | 72.6% | 41.5 | 256 | 1 | 100% |
| 6 | 2772 | 32 | 64 | 1 | 1 | 2 | 16x16x32 | 39.67 | 72.6% | 41.6 | 512 | 2 | 100% |
| 7 | 2186 | 32 | 64 | 2 | 1 | 1 | 32x32x16 | 40.00 | 73.2% | 46.6 | 512 | 2 | 100% |
| 8 | 1279 | 32 | 128 | 8 | 2 | 4 | 16x16x32 | 40.67 | 74.4% | 51.5 | 256 | 1 | 100% |
| 9 | 2376 | 64 | 32 | 1 | 1 | 2 | 16x16x32 | 40.67 | 74.4% | 41.7 | 512 | 2 | 100% |
| 10 | 2873 | 64 | 32 | 2 | 2 | 1 | 16x16x32 | 40.67 | 74.4% | 43.7 | 512 | 2 | 100% |

## Reproducing

### Compile a tuned config

```bash
iree-compile mlir/512/template.mlir \
  --compile-from=executable-sources \
  --iree-codegen-tuning-spec-path=mlir/512/1058_spec.mlir \
  --iree-hal-target-device=hip \
  --iree-rocm-target=gfx950 \
  --iree-llvmgpu-use-direct-load \
  -o output.vmfb
```

### Compile a heuristic config

```bash
iree-compile mlir/512/template.mlir \
  --compile-from=executable-sources \
  --iree-hal-target-device=hip \
  --iree-rocm-target=gfx950 \
  --iree-llvmgpu-use-direct-load \
  --iree-codegen-enable-default-tuning-specs=false \
  -o heuristic.vmfb
```

### Benchmark

```bash
iree-benchmark-module --device=hip --module=output.vmfb --benchmark_repetitions=3
```

### Generate ATT thread traces

```bash
rocprofv3 --att \
  --att-library-path /path/to/lib/ \
  --att-shader-engine-mask 0xF \
  --att-consecutive-kernels 10 \
  -d trace_output/ \
  -- iree-benchmark-module --device=hip --module=output.vmfb --benchmark_repetitions=3
```

