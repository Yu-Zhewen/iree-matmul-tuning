module attributes {iree_codegen.tuning_spec_with_default_entrypoint, transform.with_named_sequence} {
  transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield 
  }
  transform.named_sequence @match_mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %arg0, lhs_type = bf16, rhs_type = bf16, output_type = f32, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch_dims, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m_dims, [512] : !transform.param<i64>
    transform.iree.match.dims_equal %n_dims, [512] : !transform.param<i64>
    transform.iree.match.dims_equal %k_dims, [512] : !transform.param<i64>
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_BF16>, promote_operands = [0, 1], promotion_types = [#iree_gpu.use_global_load_dma, #iree_gpu.use_global_load_dma], reduction = [0, 0, 1], subgroup = [1, 4, 0], workgroup = [16, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    %updated_root = transform.foreach_match in %arg0 
        @match_mm_512$async_dispatch_0_matmul_512x512x512_bf16xbf16xf32 -> @apply_op_config : (!transform.any_op) -> !transform.any_op
    transform.yield %updated_root : !transform.any_op
  }
}