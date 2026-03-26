import random
import numpy as np
import torch

# Seed všech relevantních knihoven
def set_seed(seed):
    print(f"Set SEED: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_worker_seed_fn(seed):
    """
    Create a worker seed function for DataLoader.
    :param seed:
    :return:
    """
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker

def get_generator(seed):
    """
    Create a generator for random number generation.
    :param seed:
    :return:
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g

