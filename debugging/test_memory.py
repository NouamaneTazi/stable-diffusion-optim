from torchvision.models import resnet34, resnet50, resnet101
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.backends.xnnpack
import time

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

print("XNNPACK is enabled: ", torch.backends.xnnpack.enabled, "\n")

N, C, H, W = 1, 3, 200, 200
x = torch.rand(N, C, H, W)
print("Contiguous shape: ", x.shape)
print("Contiguous stride: ", x.stride())
print()

xcl = x.to(memory_format=torch.channels_last)
print("Channels-Last shape: ", xcl.shape)
print("Channels-Last stride: ", xcl.stride())

# m = resnet34(pretrained=False)
# m = resnet50(pretrained=False)
m = resnet101(pretrained=False)


def get_optimized_model(mm):
    mm = mm.eval()
    scripted = torch.jit.script(mm)
    optimized = optimize_for_mobile(scripted)  # explicitly call the xnnpack rewrite
    return mm, scripted, optimized


def compare_contiguous_CL(mm):
    # inference on contiguous
    start = time.perf_counter()
    for i in range(20):
        mm(x)
    end = time.perf_counter()
    print("Contiguous: ", end - start)

    # inference on channels-last
    start = time.perf_counter()
    for i in range(20):
        mm(xcl)
    end = time.perf_counter()
    print("Channels-Last: ", end - start)


with torch.inference_mode():
    m, scripted, optimized = get_optimized_model(m)

    # warmup
    for i in range(20):
        m(x)
    for i in range(20):
        scripted(x)
    for i in range(20):
        optimized(x)

    print("Runtimes for original model: ")
    compare_contiguous_CL(m.eval())
    print()
    print("Runtimes for torchscripted model: ")
    compare_contiguous_CL(scripted.eval())
    print()
    print("Runtimes for mobile-optimized model: ")
    compare_contiguous_CL(optimized.eval())
    print('=' * 80)

    print("Runtimes for original model: ")
    compare_contiguous_CL(m.eval())
    print()
    print("Runtimes for torchscripted model: ")
    compare_contiguous_CL(scripted.eval())
    print()
    print("Runtimes for mobile-optimized model: ")
    compare_contiguous_CL(optimized.eval())
    print('=' * 80)

    print("Runtimes for original model: ")
    compare_contiguous_CL(m.eval())
    print()
    print("Runtimes for torchscripted model: ")
    compare_contiguous_CL(scripted.eval())
    print()
    print("Runtimes for mobile-optimized model: ")
    compare_contiguous_CL(optimized.eval())
