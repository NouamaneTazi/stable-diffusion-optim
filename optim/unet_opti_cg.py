from tracemalloc import start
from diffusers import UNet2DConditionModel
import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import joblib
import time
import torch

latent_model_input = joblib.load('latent_model_input.pkl')
t = joblib.load('t.pkl')
text_embeddings = joblib.load('text_embeddings.pkl')

unet_path = '/home/nouamane_huggingface_co/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-3/snapshots/c0399c1dac67eb30c20b40886872cee2fdf2e6b6/unet'
unet = UNet2DConditionModel.from_pretrained(unet_path).cuda()


# warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    with torch.no_grad():
        with torch.autocast("cuda"):
            for _ in range(3):
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    with torch.no_grad():
        with torch.autocast("cuda"):
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

print(noise_pred.var())
start_time = time.time()
for _ in range(10):
    g.replay()
print(noise_pred.var())
print(noise_pred.var())
print(f"Pipeline inference took {time.time() - start_time} seconds")
time.sleep(2)
print(noise_pred.var())