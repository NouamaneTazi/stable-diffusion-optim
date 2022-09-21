from diffusers import UNet2DConditionModel
import datetime
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import joblib
import time
import torch


# test and see if these give better performance
torch._C._jit_set_nvfuser_single_node_mode(True)
# torch._C._jit_set_nvfuser_horizontal_mode(True)
# torch._C._jit_set_nvfuser_guard_mode(False) 

# CUDA VISIBLE DEVICE 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# torch disable grad
torch.set_grad_enabled(False)

torch.backends.cudnn.benchmark = True


unet_path = '/home/nouamane/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-3/snapshots/c0399c1dac67eb30c20b40886872cee2fdf2e6b6/unet'
# unet_path = '/home/nouamane_huggingface_co/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-3/snapshots/3bcaa468131c963401aa5175a14b13912b9f1933/unet' # fp16
unet = UNet2DConditionModel.from_pretrained(unet_path).half().cuda()



print(unet.conv_out.state_dict()['weight'].stride()) # (2880, 9, 3, 1)
unet.to(memory_format=torch.channels_last) # in-place operation
print(unet.conv_out.state_dict()['weight'].stride()) # (2880, 1, 960, 320) haveing a stride of 1 for the 2nd dimension proves that it works




import joblib
sample = joblib.load("sample.pkl").half()
timestep = joblib.load("timestep.pkl").half()
encoder_hidden_states = joblib.load("encoder_hidden_states.pkl").half()



# warmup
for _ in range(3):
    with torch.inference_mode():
        inputs = (torch.rand_like(sample) * sample.max(), torch.rand_like(timestep)*999, torch.rand_like(encoder_hidden_states) * encoder_hidden_states.max())
        orig_output = unet(*inputs) 

# trace        
unet_traced = torch.jit.trace(unet, inputs)



# warmup
for _ in range(5):
    with torch.inference_mode():
        inputs = (torch.rand_like(sample) * sample.max(), torch.rand_like(timestep)*999, torch.rand_like(encoder_hidden_states) * encoder_hidden_states.max())
        prev = orig_output
        orig_output = unet_traced(*inputs) 


# correctness
inputs = (sample, timestep, encoder_hidden_states)
orig_output = unet(*inputs)
new_output = unet_traced(*inputs)
try:
    torch.testing.assert_allclose(orig_output[0], new_output[0])
except AssertionError as e:
    print(e)
# Mismatched elements: 4103 / 32768 (12.5%)
# Greatest absolute difference: 0.0087890625 at index (0, 2, 8, 59) (up to 0.001 allowed)
# Greatest relative difference: 54.20996441281139 at index (0, 2, 6, 62) (up to 0.001 allowed)


# benchmarking
with torch.inference_mode():
    # for _ in range(3):
    #     torch.cuda.synchronize()
    #     start_time = time.time()
    #     for _ in range(50):
    #         orig_output = unet(*inputs)
    #     torch.cuda.synchronize()
    #     print(f"unet inference took {time.time() - start_time:.2f} seconds")

    for _ in range(3):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")

# unet inference took 4.06 seconds
# unet traced inference took 3.49 seconds
# unet (w/ channels last) inference took 3.75 seconds
# unet (w/ channels last) traced inference took 3.18 seconds

# save the model
unet_traced.save("unet_traced_CL_nofloat_singlefuse.pt")


# assert False

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_unet_traced_only_fp16_CL_nofloat_singlefuse_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:
            
    # with torch.inference_mode():
    #     for _ in range(8):
    #         with record_function("unet"):
    #             torch.cuda.synchronize()
    #             start_time = time.time()
    #             orig_output = unet(*inputs)
    #             torch.cuda.synchronize()
    #             print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")

    with torch.inference_mode():
        for _ in range(8):
            with record_function("unet_traced"):
                torch.cuda.synchronize()
                start_time = time.time()
                orig_output = unet_traced(*inputs)
                torch.cuda.synchronize()
                print(f"Pipeline inference took {time.time() - start_time:.2f} seconds")
