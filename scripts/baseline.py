# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import datetime
import time
import torch
from dataclasses import dataclass

@dataclass
class UNet2DConditionOutput():
    sample: torch.FloatTensor

prompt = "a photo of an astronaut riding a horse on mars"

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torch._C._jit_set_nvfuser_single_node_mode(True)

# cudnn benchmarking
torch.backends.cudnn.benchmark = True


torch.manual_seed(1231)
torch.cuda.manual_seed(1231)

# scheduler = LMSDiscreteScheduler(
#     beta_start=0.00085, 
#     beta_end=0.012, 
#     beta_schedule="scaled_linear"
# )

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    # scheduler=scheduler,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16
).to("cuda")

# use jitted unet
unet_traced = torch.jit.load("unet_traced.pt")
# del pipe.unet
class TracedUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = pipe.unet.in_channels
        self.device = pipe.unet.device
    def forward(self, latent_model_input, t, encoder_hidden_states):
        sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

pipe.unet = TracedUNet()


# warmup
with torch.inference_mode():
    image = pipe([prompt]*3, num_inference_steps=5).images[0]

for _ in range(3):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        image = pipe([prompt]*3, num_inference_steps=50).images[0]
    torch.cuda.synchronize()
    print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")

image.save(f"pics/pic_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
