import os
import time


def generate_seed():
    seed = os.getpid() + int(time.time() * 1e5 % 1e6)
    return seed
