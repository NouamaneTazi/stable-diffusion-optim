# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, PNDMScheduler
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime
import time
import torch

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-3", 
    scheduler=scheduler,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16
).to("cuda")

print(pipe.unet.conv_out.state_dict()['weight'].stride())
pipe.unet.to(memory_format=torch.channels_last)
print(pipe.unet.conv_out.state_dict()['weight'].stride())

prompt = "a photo of an astronaut riding a horse on mars"

# warmup
image = pipe(prompt, num_inference_steps=8)["sample"][0]  

for _ in range(3):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        image = pipe(prompt, num_inference_steps=50)["sample"][0]  
    torch.cuda.synchronize()
    print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")
image.save(f"pics/pic_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_pt_fp16_CL_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        record_shapes=True,
        profile_memory=False,
        with_stack=True
        ) as prof:

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        image = pipe(prompt, num_inference_steps=8)["sample"][0]  
    torch.cuda.synchronize()
    print(f"Pipeline inference took (w/ Profiler) {time.time() - start_time:.2f} seconds")
        