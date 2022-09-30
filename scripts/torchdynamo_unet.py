from contextlib import nullcontext
import torch
import torchdynamo
from diffusers import UNet2DConditionModel
import logging
import time
import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

# backend = "eager" # 3.10s
# backend = "aot_eager" # 3.10
# backend = "aot_cudagraphs" # 3.44s
# backend = "aot_nvfuser"
# backend = "nvfuser" # 3.02s
# backend = "inductor" # 3.28s

# torchdynamo        
# import torch
# import torchdynamo

# torchdynamo.config.verbose = True
# # torchdynamo.config.dynamic_shapes = True

# # def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
# #     print("my_compiler() called with FX graph:")
# #     gm.graph.print_tabular()
# #     return gm.forward  # return a python callable
# print("torchdynamo..")
# #debugging
# backend = "eager" # 3.15s
# backend = "aot_eager" # 3.15s
# # inference
# backend = "ofi" # 3.13s
# backend = "fx2trt"
# backend = "onnxrt"
# backend = "ipex" # CPU only
# backend = "nvfuser" # 3.06
# # backend = "nvfuser_ofi" # 3.12
# # backend = "cudagraphs" # 3.14
# backend = "cudagraphs_ts" # 3.08
# # backend = "cudagraphs_ts_ofi" # 
backend = "inductor" # 
# # backend = "static_runtime" #
# # backend = "nnc" # 3.09
# print("backend:", backend)
# unet_traced = torchdynamo.optimize(backend)(unet)
# # print("done torchdynamo")
# # unet_traced = torchdynamo.optimize("nvfuser")(unet)
# # unet_traced = torchdynamo.optimize("cudagraphs_ts")(unet_traced)
# unet_traced.eval()

torchdynamo.config.HAS_REFS_PRIMS = True
torchdynamo.config.capture_scalar_outputs = False
torchdynamo.config.dead_code_elimination = True
torchdynamo.config.dynamic_propagation = True
torchdynamo.config.dynamic_shapes = False
torchdynamo.config.enforce_cond_guards_match = True
torchdynamo.config.fake_tensor_propagation = True
torchdynamo.config.guard_nn_modules = False
torchdynamo.config.normalize_ir = False
torchdynamo.config.optimize_ddp = False
torchdynamo.config.raise_on_assertion_error = False
torchdynamo.config.raise_on_backend_error = True
torchdynamo.config.raise_on_ctx_manager_usage = True
torchdynamo.config.specialize_int_float = True
torchdynamo.config.verbose = False
torchdynamo.config.verify_correctness = False
# torchdynamo.config.log_level = logging.DEBUG

torch.set_grad_enabled(False)

import json
config = json.load(open("unet.json"))

unet = UNet2DConditionModel(**config).half().cuda()
unet.to(memory_format=torch.channels_last)
unet.eval()
unet_traced = torchdynamo.optimize(backend)(unet)

sample = torch.randn(2, 4, 64, 64).half().cuda()
timestep = torch.rand(1).half().cuda() * 999
encoder_hidden_states = torch.randn(2, 77, 768).half().cuda()
inputs = (sample, timestep, encoder_hidden_states)

unet_traced(sample, timestep, encoder_hidden_states)
print("done")

# benchmarking
# with torch.inference_mode():
with nullcontext():
    for _ in range(3):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"{backend}: unet traced inference took {time.time() - start_time:.2f} seconds")
    # for _ in range(2):
    #     torch.cuda.synchronize()
    #     start_time = time.time()
    #     for _ in range(50):
    #         orig_output = unet(*inputs)
    #     torch.cuda.synchronize()
    #     print(f"unet inference took {time.time() - start_time:.2f} seconds")

# eager: unet inference took 3.40 seconds
# nvfuser: unet traced inference took 3.35 seconds


# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_unet_traced_only_{backend}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
#         ) as prof:
            
#     with torch.inference_mode():
#         torch.cuda.synchronize()
#         start_time = time.time()
#         for _ in range(8):
#             with record_function("unet_traced"):
#                 orig_output = unet_traced(*inputs)
#         torch.cuda.synchronize()
#         print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")
