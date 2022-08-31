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
unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=None).cuda()

with torch.no_grad():
    with torch.autocast("cuda"):
        start_time = time.time()
        for _ in range(10):
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
        print(f"Pipeline inference took {time.time() - start_time} seconds")


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_unet_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        record_shapes=True,
        profile_memory=False,
        with_stack=True
        ) as prof:

    with torch.no_grad():
        with torch.autocast("cuda"):
            for _ in range(10):
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
