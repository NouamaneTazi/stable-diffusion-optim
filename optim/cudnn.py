# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime
import time

#CUDA AVAILABLE DEVICES
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#cudnn benchmark
import torch
torch.backends.cudnn.benchmark = True

lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-3", 
    scheduler=lms,
    use_auth_token=True
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"

# warmup
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]  

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_pt_cudnn_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        record_shapes=False,
        profile_memory=False,
        with_stack=False
        ) as prof:

    start_time = time.time()
    with autocast("cuda"):
        image = pipe(prompt)["sample"][0]  
    print(image)
    print(f"Pipeline inference took {time.time() - start_time} seconds")
        

image.save("astronaut_rides_horse.png")