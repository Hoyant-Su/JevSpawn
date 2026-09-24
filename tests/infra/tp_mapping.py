import os
import time

import torch


print({key: os.environ.get(key) for key in ['LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES']},
      torch.cuda.current_device(), os.getpid(), flush=True)
time.sleep(20)
