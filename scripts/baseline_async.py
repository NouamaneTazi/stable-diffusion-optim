# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import datetime
import time
import torch
import os
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime
prompt = "a photo of an astronaut riding a horse on mars"
import torchvision.transforms as T

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch._C._jit_set_nvfuser_single_node_mode(True)

# cudnn benchmarking
# torch.backends.cudnn.benchmark = True
BATCH_SIZE = 1

torch.manual_seed(1231)
torch.cuda.manual_seed(1231)


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


# warmup
with torch.inference_mode():
    image = pipe([prompt]*BATCH_SIZE, num_inference_steps=5, return_async=True)

for _ in range(3):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        image = pipe([prompt]*BATCH_SIZE, num_inference_steps=50, return_async=True)
    torch.cuda.synchronize()
    print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")

# image = image.cpu().permute(0, 2, 3, 1).numpy()
transform = T.ToPILImage()
transform(image[0]).save(f"pics/pic_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_pt_fp16_old_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        image = pipe([prompt]*BATCH_SIZE, num_inference_steps=8, return_async=True)
    torch.cuda.synchronize()
    print(f"Pipeline inference took (w/ Profiler) {time.time() - start_time:.2f} seconds")