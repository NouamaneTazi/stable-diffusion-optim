from contextlib import nullcontext
import torch
import torchdynamo
from diffusers import UNet2DConditionModel
import logging
import time
import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

# backend = "eager" # 3.10s
# backend = "aot_eager" # 3.10
backend = "aot_cudagraphs" # 3.44s
# backend = "aot_nvfuser"
# backend = "nvfuser" # 3.02s
# backend = "cudagraphs" # 3.14
# backend = "cudagraphs_ts" # 3.08

torchdynamo.config.verbose = True
# torchdynamo.config.log_level = logging.DEBUG

torch.set_grad_enabled(False)

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    scheduler=scheduler,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16
).to("cuda")

@torchdynamo.optimize(backend)
# @torchdynamo.optimize(backend, nopython=True)
def pipe_opt(prompt):
    return pipe([prompt]*1, num_inference_steps=8).images[0]

prompt = "a photo of an astronaut riding a horse on mars"

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:
    pipe_opt(prompt)
print("done")
for _ in range(2):
    image = pipe_opt(prompt)

# benchmarking
# with torch.inference_mode():
with nullcontext():
    for _ in range(3):
        torch.cuda.synchronize()
        start_time = time.time()
        image = pipe_opt(prompt)
        torch.cuda.synchronize()
        print(f"{backend}: SD pipeline inference took {time.time() - start_time:.2f} seconds")
image.save(f"pics/pic_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

# eager: SD pipeline inference took 3.91 seconds

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_pipe_opt_{backend}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:
            
    with nullcontext():
        torch.cuda.synchronize()
        start_time = time.time()
        image = pipe_opt(prompt)
        torch.cuda.synchronize()
        print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")
