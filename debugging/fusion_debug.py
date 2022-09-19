# https://github.com/pytorch/pytorch/blob/release/1.12/torch/csrc/jit/codegen/cuda/README.md

# PYTORCH_NVFUSER_DISABLE=fallback PYTORCH_NVFUSER_DUMP=perf_debug_verbose python debugging/fusion_debug.py 
# PYTORCH_NVFUSER_DUMP=dump_eff_bandwidth python debugging/fusion_debug.py 
# PYTORCH_JIT_LOG_LEVEL=graph_fuser python debugging/fusion_debug.py
# Available options:
        # fusion_ir, fusion_ir_math, kernel_ir, ca_map, cuda_kernel, cuda_full,
        # cuda_to_file, launch_param, segmented_fusion, fusion_args,
        # kernel_args, dump_eff_bandwidth, draw_segmented_fusion,
        # scheduler_params, parallel_dimensions, buffer_reuse_verbose,
        # ptxas_verbose, halo, segmenter_logging, perf_debug_verbose

import torch

def forward(x):
    o = x + 1.0
    o = o.relu()
    return o

input = torch.rand((2, 32, 128, 512)).cuda()

t = torch.jit.script(forward)
print(t.graph_for(input))

# t = torch.jit.script(forward)
# with torch.jit.fuser("fuser0"):
#     for k in range(4):
#         o = t(input)
# print("fuser0")
# print(t.graph_for(input))

# t = torch.jit.script(forward)
# with torch.jit.fuser("fuser1"):
#     for k in range(4):
#         o = t(input)
# print("fuser1")
# print(t.graph_for(input))

t = torch.jit.script(forward)
with torch.jit.fuser("fuser2"):
    for k in range(4):
        o = t(input)
print("fuser2")
print(t.graph_for(input))



print()