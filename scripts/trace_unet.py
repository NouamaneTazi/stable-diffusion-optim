import time
from typing import List
import torch
from diffusers import StableDiffusionPipeline
import functools


# test and see if these give better performance
# torch._C._jit_set_nvfuser_single_node_mode(True)
# torch._C._jit_set_nvfuser_horizontal_mode(True)
# torch._C._jit_set_nvfuser_guard_mode(False) 

# CUDA VISIBLE DEVICE 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# torch disable grad
torch.set_grad_enabled(False)

# load inputs
BATCH_SIZE = 1
def generate_inputs():
    sample = torch.randn(BATCH_SIZE*2, 4, 64, 64).half().cuda()
    timestep = torch.rand(1).half().cuda() * 999
    encoder_hidden_states = torch.randn(BATCH_SIZE*2, 77, 768).half().cuda()
    return sample, timestep, encoder_hidden_states


pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    # scheduler=scheduler,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16
).to("cuda")
unet = pipe.unet
unet.eval()
unet.to(memory_format=torch.channels_last) # use channels_last memory format
# unet.forward = functools.partial(unet.forward, return_dict=False) # set return_dict=False as default

# warmup
for _ in range(3):
    inputs = generate_inputs()
    orig_output = unet(*inputs)

# trace        
# print("tracing..")
# unet_traced = torch.jit.trace(unet, inputs)
# unet_traced = torch.jit.optimize_for_inference(unet_traced)
# unet_traced.eval()
# print("done tracing")

# torchdynamo        
import torch
import torchdynamo

torchdynamo.config.verbose = True
# torchdynamo.config.dynamic_shapes = True

# def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
#     print("my_compiler() called with FX graph:")
#     gm.graph.print_tabular()
#     return gm.forward  # return a python callable
print("torchdynamo..")
#debugging
backend = "eager" # 3.15s
backend = "aot_eager" # 3.15s
# inference
backend = "ofi" # 3.13s
backend = "fx2trt"
backend = "onnxrt"
backend = "ipex" # CPU only
backend = "nvfuser" # 3.06
backend = "nvfuser_ofi" # 3.12
backend = "cudagraphs" # 3.14
backend = "cudagraphs_ts" # 3.08
backend = "cudagraphs_ts_ofi" # 
backend = "inductor" # 
backend = "static_runtime" #
backend = "nnc" # 3.09
print("backend:", backend)
unet_traced = torchdynamo.optimize(backend)(unet)
# print("done torchdynamo")
# unet_traced = torchdynamo.optimize("nvfuser")(unet)
# unet_traced = torchdynamo.optimize("cudagraphs_ts")(unet_traced)
unet_traced.eval()

# warmup and optimize graph
for _ in range(5):
    inputs = generate_inputs()
    orig_output = unet_traced(*inputs) 


# benchmarking
with torch.inference_mode():
    for _ in range(3):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            orig_output = unet_traced(*inputs)
        torch.cuda.synchronize()
        print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
    for _ in range(2):
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            orig_output = unet(*inputs)
        torch.cuda.synchronize()
        print(f"unet inference took {time.time() - start_time:.2f} seconds")

# unet inference took 3.14 seconds
# unet traced inference took 3.07 seconds


# torchdynamo - 3 images
# unet inference took 8.21 seconds



# save the model
# assert False
# unet_traced.save("unet_traced.pt")

