# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime
import time
import torch

#CUDA AVAILABLE DEVICES
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(1):
        with torch.no_grad():
            with autocast("cuda"):
                image = pipe(prompt)["sample"][0]  
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    with torch.no_grad():
        with autocast("cuda"):
            image = pipe(prompt)["sample"][0]  


start_time = time.time()
# inp["input_ids"].copy_(inp["input_ids"].cuda())
# inp["attention_mask"].copy_(inp["attention_mask"].cuda())
g.replay() # replay the graph and updates outputs
print(image)
print(f"Pipeline inference took {time.time() - start_time} seconds")
    

image.save("astronaut_rides_horse.png")