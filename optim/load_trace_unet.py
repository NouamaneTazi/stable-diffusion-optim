from diffusers import UNet2DConditionModel
import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import joblib
import time
import torch


# CUDA VISIBLE DEVICE 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# torch disable grad
torch.set_grad_enabled(False)



import joblib
sample = joblib.load("sample.pkl").half()
timestep = joblib.load("timestep.pkl").half()
encoder_hidden_states = joblib.load("encoder_hidden_states.pkl").half()
inputs = (sample, timestep, encoder_hidden_states)

unet_traced = torch.jit.load("unet_traced_CL_nofloat_singlefuse.pt")


# warmup
for _ in range(5):
    with torch.inference_mode():
        orig_output = unet_traced(*inputs) 


#benchmarking
with torch.inference_mode():
    for _ in range(3):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")

# unet inference took 4.06 seconds
# unet traced inference took 3.49 seconds
# unet (w/ channels last) inference took 3.75 seconds
# unet (w/ channels last) traced inference took 3.18 seconds


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_unet_traced_only_fp16_CL_nofloat_singlefuse_loaded_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:
            
    # with torch.inference_mode():
    #     for _ in range(8):
    #         with record_function("unet"):
    #             torch.cuda.synchronize()
    #             start_time = time.time()
    #             orig_output = unet(*inputs)
    #             torch.cuda.synchronize()
    #             print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")

    with torch.inference_mode():
        for _ in range(8):
            with record_function("unet_traced"):
                torch.cuda.synchronize()
                start_time = time.time()
                orig_output = unet_traced(*inputs)
                torch.cuda.synchronize()
                print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")
