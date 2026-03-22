#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_mi355x.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx950", features = "", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [<MFMA_F32_16x16x32_F16>, <MFMA_F32_32x32x16_F16>, <MFMA_F32_16x16x32_BF16>, <MFMA_F32_32x32x16_BF16>, <MFMA_F32_16x16x128_F8E5M2>, <MFMA_F32_16x16x128_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x64_I8>, <MFMA_I32_32x32x32_I8>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2>, <MFMA_F32_16x16x32_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], scaled_mma = [<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E5M2, rhs_elem_type = f8E5M2, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E5M2FNUZ, rhs_elem_type = f8E5M2FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FNUZ, rhs_elem_type = f8E4M3FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E5M2, rhs_elem_type = f8E5M2, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E5M2FNUZ, rhs_elem_type = f8E5M2FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E4M3FNUZ, rhs_elem_type = f8E4M3FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 163840, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384, dma_sizes = [32, 128], workgroup_memory_bank_count = 64>, chip = <wgp_count = 256, sku = "mi355x", memory_bandwidth_tbps = 8.000000e+00 : f32, perf_tflops = {fp16 = 2.500000e+03 : f32, fp32 = 1.573000e+02 : f32, fp4 = 1.000000e+04 : f32, fp6 = 1.000000e+04 : f32, fp8 = 5.000000e+03 : f32, int8 = 5.000000e+03 : f32}>>, ukernels = "none"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#device_target_hip = #hal.device.target<"hip", [#executable_target_rocm_hsaco_fb]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_hip
  hal.executable private @mm_8192$async_dispatch_0 {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @mm_8192$async_dispatch_0_matmul_8192x8192x8192_bf16xbf16xf32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @mm_8192$async_dispatch_0_matmul_8192x8192x8192_bf16xbf16xf32() {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xbf16>>
          %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xbf16>>
          %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192x8192xbf16>>
          %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xbf16>> -> tensor<8192x8192xbf16>
          %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xbf16>> -> tensor<8192x8192xbf16>
          %5 = tensor.empty() : tensor<8192x8192xbf16>
          %6 = tensor.empty() : tensor<8192x8192xf32>
          %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
          %8 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%3, %4 : tensor<8192x8192xbf16>, tensor<8192x8192xbf16>) outs(%7 : tensor<8192x8192xf32>) -> tensor<8192x8192xf32>
          %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<8192x8192xf32>) outs(%5 : tensor<8192x8192xbf16>) {
          ^bb0(%in: f32, %out: bf16):
            %10 = arith.truncf %in : f32 to bf16
            linalg.yield %10 : bf16
          } -> tensor<8192x8192xbf16>
          iree_tensor_ext.dispatch.tensor.store %9, %2, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : tensor<8192x8192xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192x8192xbf16>>
          return
        }
      }
    }
  }
  util.func public @mm_8192$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.fence, %arg3: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %c0 = arith.constant 0 : index
    %c134217728 = arith.constant 134217728 : index
    %c8192 = arith.constant 8192 : index
    %element_type_bf16 = hal.element_type<bf16> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("tensor") shape([%c8192, %c8192]) type(%element_type_bf16) encoding(%dense_row_major)
    %0 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<8192x8192xbf16> in !stream.resource<external>{%c134217728}
    %1 = stream.timepoint.import on(#hal.device.affinity<@__device_0>) %arg2 : (!hal.fence) => !stream.timepoint
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("tensor") shape([%c8192, %c8192]) type(%element_type_bf16) encoding(%dense_row_major)
    %2 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<8192x8192xbf16> in !stream.resource<external>{%c134217728}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) await(%1) => !stream.resource<external>{%c134217728} => !stream.timepoint
    %3 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%0 as %arg4: !stream.resource<external>{%c134217728}, %2 as %arg5: !stream.resource<external>{%c134217728}, %result as %arg6: !stream.resource<external>{%c134217728}) {
      stream.cmd.dispatch @mm_8192$async_dispatch_0::@rocm_hsaco_fb::@mm_8192$async_dispatch_0_matmul_8192x8192x8192_bf16xbf16xf32 {
        ro %arg4[%c0 for %c134217728] : !stream.resource<external>{%c134217728},
        ro %arg5[%c0 for %c134217728] : !stream.resource<external>{%c134217728},
        wo %arg6[%c0 for %c134217728] : !stream.resource<external>{%c134217728}
      }
    } => !stream.timepoint
    stream.timepoint.chain_external on(#hal.device.affinity<@__device_0>) %3 => (%arg3 : !hal.fence)
    %4 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %result : tensor<8192x8192xbf16> in !stream.resource<external>{%c134217728} -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
  util.func public @mm_8192(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @mm_8192$async(%arg0, %arg1, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.return %1 : !hal.buffer_view
  }
}
