# import conv2D
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime
import time
# Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
conv2D = nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to("cuda").half().to(memory_format=torch.channels_last)
# random torch.Size([2, 320, 64, 64])
hidden_stats = torch.rand(2, 320, 64, 64, device="cuda").half()


for _ in range(3):
    with torch.inference_mode():
        out = conv2D(hidden_stats)


# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_conv_fp16_CL_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
#         record_shapes=True,
#         profile_memory=False,
#         with_stack=True
#         ) as prof:

#     torch.cuda.synchronize()
#     start_time = time.time()
#     for _ in range(3):
#         with torch.inference_mode():
#             out = conv2D(hidden_stats)
#     torch.cuda.synchronize()
#     print(f"Pipeline inference took (w/ Profiler) {time.time() - start_time:.2f} seconds")