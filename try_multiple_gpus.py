import torch.multiprocessing as mp

def f(rank, arg):
    print(f"Process {rank}: Hello World (arg={arg})")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # Force Colab-compatible start method
    x = 1
    mp.spawn(f, args=(x,), nprocs=1, join=True)