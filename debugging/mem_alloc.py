# import conv2D
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime
import time
from pprint import pprint


# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_mem_alloc_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
#         ) as prof:
torch.cuda.synchronize()
start_time = time.time()
for i in range(10):
    A = torch.rand(i, 99999).to("cuda")
torch.cuda.synchronize()
print(f"A took {time.time() - start_time} seconds")

torch.cuda.synchronize()
start_time = time.time()
for i in range(10):
    B = torch.rand(10, 99999).to("cuda")
torch.cuda.synchronize()
print(f"B took {time.time() - start_time} seconds")

pprint(torch.cuda.memory_stats_as_nested_dict())
