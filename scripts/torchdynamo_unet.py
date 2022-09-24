import torch
import torchdynamo
from diffusers import UNet2DConditionModel
import logging
import time
import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

backend = "inductor_cudagraphs"

torchdynamo.config.verbose = True
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
with torch.inference_mode():
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
