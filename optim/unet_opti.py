from tracemalloc import start
from diffusers import UNet2DConditionModel
import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import joblib
import time
import torch

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.benchmark = True
# torch.manual_seed(123154)
# torch.cuda.manual_seed(123154)

unet_path = '/home/nouamane/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-3/snapshots/c0399c1dac67eb30c20b40886872cee2fdf2e6b6/unet'
# unet_path = '/home/nouamane_huggingface_co/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-3/snapshots/3bcaa468131c963401aa5175a14b13912b9f1933/unet' # fp16
unet = UNet2DConditionModel.from_pretrained(unet_path).cuda().to(memory_format=torch.channels_last)

with torch.no_grad():
    with torch.autocast("cuda"):
        for _ in range(2):
            latent_model_input = joblib.load('latent_model_input.pkl')[:4].repeat(2, 1, 1, 1).to(memory_format=torch.channels_last)
            t = joblib.load('t.pkl')
            text_embeddings = joblib.load('text_embeddings.pkl')[:4].repeat(2, 1, 1)[:,:72,:]
            start_time = time.time()
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
            print(noise_pred.var())
            print(f"Pipeline inference took {time.time() - start_time} seconds")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_unet_autocast_8_72_cudnn_CL_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        record_shapes=True,
        profile_memory=False,
        with_stack=True
        ) as prof:
    with torch.no_grad():
        with torch.autocast("cuda"):
            for _ in range(5):
                latent_model_input = joblib.load('latent_model_input.pkl')[:4].repeat(2, 1, 1, 1).to(memory_format=torch.channels_last)
                t = joblib.load('t.pkl')
                text_embeddings = joblib.load('text_embeddings.pkl')[:4].repeat(2, 1, 1)[:,:72,:]
                start_time = time.time()
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
                print(noise_pred.var())
                print(f"Pipeline inference took {time.time() - start_time} seconds")
