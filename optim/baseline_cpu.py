# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, PNDMScheduler
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime
import time
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
)

prompt = "a photo of an astronaut riding a horse on mars"

# warmup
image = pipe(prompt, num_inference_steps=8)["sample"][0]  

for _ in range(3):
    start_time = time.time()
    image = pipe(prompt, num_inference_steps=8)["sample"][0]  
    print(image)
    print(f"Pipeline inference took {time.time() - start_time} seconds")
image.save(f"pics/pic_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_pt_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
#         record_shapes=True,
#         profile_memory=False,
#         with_stack=True
#         ) as prof:

#     start_time = time.time()
#     with autocast("cuda"):
#         image = pipe(prompt, num_inference_steps=4)["sample"][0]  
#     print(image)
#     print(f"Pipeline inference took (w/ Profiler) {time.time() - start_time} seconds")
        
# Before:
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:53<00:00,  6.63s/it]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:53<00:00,  6.74s/it]
# <PIL.Image.Image image mode=RGB size=512x512 at 0x7F6BFAF4EE50>
# Pipeline inference took 59.32412338256836 seconds
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:54<00:00,  6.80s/it]
# <PIL.Image.Image image mode=RGB size=512x512 at 0x7F6BFAF4EC10>

# Using CL
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:53<00:00,  6.63s/it]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:53<00:00,  6.70s/it]
# <PIL.Image.Image image mode=RGB size=512x512 at 0x7EFE72D2D880>
# Pipeline inference took 59.314711809158325 seconds
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:54<00:00,  6.76s/it]
# <PIL.Image.Image image mode=RGB size=512x512 at 0x7EFE72D2DD30>
# Pipeline inference took 59.795530557632446 seconds