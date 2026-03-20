#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.target_info = #iree_gpu.target<arch = "gfx950", features = "", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [<MFMA_F32_16x16x32_F16>, <MFMA_F32_32x32x16_F16>, <MFMA_F32_16x16x32_BF16>, <MFMA_F32_32x32x16_BF16>, <MFMA_F32_16x16x128_F8E5M2>, <MFMA_F32_16x16x128_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN>, <MFMA_F32_16x16x128_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2>, <MFMA_F32_32x32x64_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN>, <MFMA_F32_32x32x64_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x64_I8>, <MFMA_I32_32x32x32_I8>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2>, <MFMA_F32_16x16x32_F8E5M2_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN>, <MFMA_F32_16x16x32_F8E4M3FN_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2>, <MFMA_F32_32x32x16_F8E5M2_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN>, <MFMA_F32_32x32x16_F8E4M3FN_F8E5M2>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], scaled_mma = [<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E5M2, rhs_elem_type = f8E5M2, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E5M2FNUZ, rhs_elem_type = f8E5M2FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FNUZ, rhs_elem_type = f8E4M3FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E5M2, rhs_elem_type = f8E5M2, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E5M2FNUZ, rhs_elem_type = f8E5M2FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E4M3FNUZ, rhs_elem_type = f8E4M3FNUZ, acc_elem_type = f32>, <intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 163840, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384, dma_sizes = [32, 128], workgroup_memory_bank_count = 64>>, ukernels = "none"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#device_target_hip = #hal.device.target<"hip", [#executable_target_rocm_hsaco_fb]> : !hal.device
module {
  util.global private @__device_0 = #device_target_hip
  hal.executable private @mm_512$async_dispatch_0 {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32() {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<512x512xbf16, #hal.descriptor_type<storage_buffer>>
          %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<512x512xbf16, #hal.descriptor_type<storage_buffer>> to memref<512x512xbf16, #amdgpu.address_space<fat_raw_buffer>>
          %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<512x512xbf16, #hal.descriptor_type<storage_buffer>>
          %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<512x512xbf16, #hal.descriptor_type<storage_buffer>> to memref<512x512xbf16, #amdgpu.address_space<fat_raw_buffer>>
          %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<512x512xbf16, #hal.descriptor_type<storage_buffer>>
          %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<512x512xbf16, #hal.descriptor_type<storage_buffer>> to memref<512x512xbf16, #amdgpu.address_space<fat_raw_buffer>>
          %6 = iree_codegen.load_from_buffer %1 : memref<512x512xbf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<512x512xbf16>
          %7 = iree_codegen.load_from_buffer %3 : memref<512x512xbf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<512x512xbf16>
          %8 = tensor.empty() : tensor<512x512xbf16>
          %9 = tensor.empty() : tensor<512x512xf32>
          %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<512x512xf32>) -> tensor<512x512xf32>
          %11 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6, %7 : tensor<512x512xbf16>, tensor<512x512xbf16>) outs(%10 : tensor<512x512xf32>) {
          ^bb0(%in: bf16, %in_0: bf16, %out: f32):
            %13 = arith.extf %in : bf16 to f32
            %14 = arith.extf %in_0 : bf16 to f32
            %15 = arith.mulf %13, %14 : f32
            %16 = arith.addf %out, %15 : f32
            linalg.yield %16 : f32
          } -> tensor<512x512xf32>
          %12 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<512x512xf32>) outs(%8 : tensor<512x512xbf16>) {
          ^bb0(%in: f32, %out: bf16):
            %13 = arith.truncf %in : f32 to bf16
            linalg.yield %13 : bf16
          } -> tensor<512x512xbf16>
          iree_codegen.store_to_buffer %12, %5 : tensor<512x512xbf16> into memref<512x512xbf16, #amdgpu.address_space<fat_raw_buffer>>
          return
        }
      }
    }
  }
  util.global private mutable @mm_512$async_dispatch_0_rocm_hsaco_fb_mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32_buffer : !hal.buffer
  util.initializer {
    %device, %queue_affinity = hal.device.resolve on(#hal.device.affinity<@__device_0>) : !hal.device, i64
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
    %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
    %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
    %c1572864 = arith.constant 1572864 : index
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%queue_affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%c1572864}
    util.global.store %buffer, @mm_512$async_dispatch_0_rocm_hsaco_fb_mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @mm_512$async_dispatch_0_rocm_hsaco_fb_mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %0 = arith.index_cast %arg0 : i32 to index
    %device, %queue_affinity = hal.device.resolve on(#hal.device.affinity<@__device_0>) : !hal.device, i64
    %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) affinity(%queue_affinity) : !hal.command_buffer
    %mm_512$async_dispatch_0_rocm_hsaco_fb_mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32_buffer = util.global.load @mm_512$async_dispatch_0_rocm_hsaco_fb_mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32_buffer : !hal.buffer
    %c0 = arith.constant 0 : index
    %c524288 = arith.constant 524288 : index
    %c1048576 = arith.constant 1048576 : index
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device : !hal.device) target(@mm_512$async_dispatch_0::@rocm_hsaco_fb::@mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@mm_512$async_dispatch_0) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@mm_512$async_dispatch_0::@rocm_hsaco_fb::@mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32) : index
    %c1 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %0 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z]) bindings([
        (%mm_512$async_dispatch_0_rocm_hsaco_fb_mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32_buffer : !hal.buffer)[%c0, %c524288], 
        (%mm_512$async_dispatch_0_rocm_hsaco_fb_mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32_buffer : !hal.buffer)[%c524288, %c524288], 
        (%mm_512$async_dispatch_0_rocm_hsaco_fb_mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32_buffer : !hal.buffer)[%c1048576, %c524288]
      ]) flags("None")
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %1 = util.null : !hal.fence
    %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device : !hal.device> affinity(%queue_affinity) wait(%1) signal(%fence) commands(%cmd) flags("None")
    %c-1_i32 = arith.constant -1 : i32
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}
