# dlprof --nsys_opts="-t cuda,nvtx" --mode=pytorch --output_path=./dlprof_logs/pipeline_unet_cg python optim/pipeline_cg_dlprof.py

# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, PNDMScheduler
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime
import time
import torch
#CUDA AVAILABLE DEVICES
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-3", 
    scheduler=scheduler,
    use_auth_token=True,
    # revision="fp16",
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"

# warmup

torch.cuda.nvtx.range_push("warmup")
with autocast("cuda"):
    image = pipe([prompt]*2, num_inference_steps=8)["sample"][0]  
torch.cuda.nvtx.range_pop()


for _ in range(3):
    torch.cuda.nvtx.range_push("inference")
    start_time = time.time()
    with autocast("cuda"):
        image = pipe([prompt]*2, num_inference_steps=50)["sample"][0]  
    print(image)
    print(f"Pipeline inference took {time.time() - start_time} seconds")
    torch.cuda.nvtx.range_pop()

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_pt_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
#         record_shapes=False,
#         profile_memory=False,
#         with_stack=False
#         ) as prof:

#     start_time = time.time()
#     with autocast("cuda"):
#         image = pipe(prompt, num_inference_steps=8)["sample"][0]  
#     print(image)
#     print(f"Pipeline inference took (w/ Profiler) {time.time() - start_time} seconds")
        
image.save(f"pics/pic_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")