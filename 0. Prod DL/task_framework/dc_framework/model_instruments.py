import logging
import numpy as np

import torch

from pathlib import Path
from typing import Dict

from dc_framework.data_preparation import Dataset

logger = logging.getLogger("__name__")


def init(model: torch.nn.Module, criterion: torch.nn.Module):
    if torch.cuda.device_count() > 1
      torch.distributed.init_process_group(
          "nccl",
          rank=torch.distributed.get_rank(),
          world_size=torch.cuda.device_count()
      )
    return DCFramework(model, criterion)


class DCFramework:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, lr=1e-3, on_cuda: bool = False):
        self.gpuc = torch.cuda.device_count()
        self.model = model.cuda() if on_cuda else model.cpu()
        if self.gpuc > 1:
	  model = torch.nn.parallel.DistributedDataParallel(model)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.criterion = criterion
        self.on_cuda = on_cuda

    def forward(self, feature, target):
        try:
            output = self.model(feature)
        except:
            logger.warning(f"feature: {feature}")
            raise
        try:
            loss = self.criterion(output, target)
        except:
            logger.warning(f"output: {output}")
            logger.warning(f"target: {target}")
            raise
        return {
            "output": output,
            "loss": loss
        }

    def train_step(self, batch, batch_indx):
        device = device = f"cuda:{torch.distributed.get_rank()}" if self.on_cuda else "cpu"
        output = self.forward(*batch.to(device))
        loss = output["loss"]
        loss.backward()
        self.optimizer.step()

    def validate_step(self, batch, batch_indx):
        device = device = f"cuda:{torch.distributed.get_rank()}" if self.on_cuda else "cpu"
        output = self.forward(*batch.to(device))
        loss = output["loss"]

    def test_step(self, batch, batch_indx):
        self.validate_step(batch, batch_indx)

    def train(self, train_data: Dict[str, np.array], batch_size: int = 1, validate_data: Dict[str, np.array] = None,
              validation_rarity: int = 20):
        train_data = Dataset(train_data)
        train_dataloader = train_data.get_dataloader(batch_size=batch_size)
        if validate_data is not None:
            validate_data = Dataset(validate_data)
            validate_dataloader = validate_data.get_dataloader(batch_size=batch_size)
        for i, batch in enumerate(train_dataloader):
            self.train_step(batch, batch_indx=i)
            if (i + 1) % validation_rarity == 0:
                with torch.no_grad():
                    for j, v_batch in enumerate(validate_dataloader):
                        self.validate_step(batch, batch_indx=j)

    def validate(self, validate_data: Dict[str, np.array], batch_size: int = 1):
        validate_data = Dataset(validate_data)
        validate_dataloader = validate_data.get_dataloader(batch_size=batch_size)
        with torch.no_grad():
            for batch in validate_dataloader:
                output = self.forward(*batch)
                loss = output["loss"]

    def test(self, test_data: Dict[str, np.array], batch_size: int = 1):
        self.validate(test_data, batch_size)

    def load_on_gpu(self):
        if self.on_cuda:
            pass
        self.model.cuda()

    def load_on_cpu(self):
        if not self.on_cuda:
            pass
        self.model.cpu()

    def save(self, path: Path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: Path):
        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
