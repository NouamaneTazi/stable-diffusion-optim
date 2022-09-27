import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import datetime

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(f"./tb_logs/tb_non_blocking_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:

        # stream = torch.cuda.current_stream()

        x = torch.rand(32, 256, 220, 220).cuda()
        t = torch.rand(32, 256, 220, 220).cuda(1)

        x = (x.min() - x.max()).to(torch.device("cpu"), non_blocking=True)
        t = x.to("cuda:1", non_blocking=True)

        print(t) # tensor(0., device='cuda:1')
        torch.cuda.synchronize() # wait for stream to finish the work
        # t = x.to("cuda:1", non_blocking=True)
        print(t) # tensor(-1.0000, device='cuda:1')