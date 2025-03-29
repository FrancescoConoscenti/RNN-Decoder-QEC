import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
#from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank, world_size):
    
    #Args:
    #    rank: Unique identifier of each process
    #    world_size: Total number of processes
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    #torch.cuda.set_device(rank)
    init_process_group(backend="gloo", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model #.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model)# device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.mse_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            #source = source.to(self.gpu_id)
            #targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    num_samples = 128
    input_size = 20
    output_size = 2

    inputs = torch.randn(num_samples, input_size)
    targets = torch.randint(0, output_size + 1, (num_samples,)) # Changed target creation

    train_set = TensorDataset(inputs, targets)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(id, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(id, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, id, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    
    save_every=1
    total_epochs=2
    batch_size=16

    world_size = 4
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size,join=True, daemon=False, start_method='spawn')

"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def train(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Model and DDP wrapper
    model = torch.nn.Linear(10, 1)  # Example model
    ddp_model = DDP(model)
    
    # Training loop
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    for _ in range(5):
        optimizer.zero_grad()
        output = ddp_model(torch.randn(5, 10))  # Fake batch
        loss = output.sum()
        loss.backward()
        optimizer.step()
    
    print(f"Rank {rank} finished training")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4  # Number of CPU processes
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
"""