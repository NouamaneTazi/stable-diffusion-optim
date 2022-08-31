from tracemalloc import start
from diffusers import UNet2DConditionModel
import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import joblib
import time
import torch
torch.backends.cudnn.benchmark = True

latent_model_input = joblib.load('latent_model_input_8.pkl')[:8]
t = joblib.load('t_8.pkl')
text_embeddings = joblib.load('text_embeddings_8.pkl')[:8]

unet_path = '/home/nouamane_huggingface_co/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-3/snapshots/c0399c1dac67eb30c20b40886872cee2fdf2e6b6/unet'
# unet_path = '/home/nouamane_huggingface_co/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-3/snapshots/3bcaa468131c963401aa5175a14b13912b9f1933/unet' # fp16
unet = UNet2DConditionModel.from_pretrained(unet_path).cuda()

# warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    with torch.no_grad():
        for _ in range(3):
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

for _ in range(3):
    start_time = time.time()
    g.replay()
    print(noise_pred.var())
    print(f"Pipeline inference took (CG) {time.time() - start_time} seconds")

with torch.no_grad():
    for _ in range(3):
        latent_model_input = joblib.load('latent_model_input_8.pkl')[:8]
        t = joblib.load('t_8.pkl')
        text_embeddings = joblib.load('text_embeddings_8.pkl')[:8]
        start_time = time.time()
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
        print(noise_pred.var())
        print(f"Pipeline inference took {time.time() - start_time} seconds")

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_unet_CG_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
#         record_shapes=True,
#         profile_memory=False,
#         with_stack=True
#         ) as prof:

#     g.replay()
#     print(noise_pred.var())