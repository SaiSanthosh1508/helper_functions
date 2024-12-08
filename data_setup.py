import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from typing import Tuple,List,Union

def create_dataloader(dir: str,
                      transforms: transforms.Compose,
                      batch_size: int,
                      shuffle: bool = False,
                      num_workers: int = os.cpu_count(),
                      return_class_names : bool = False
                      )-> Union[DataLoader, Tuple[DataLoader, List[str]]]:
  """
  Create datalaoders for computer vision tasks utilising ImageFolder and DataLoader.

  Args:
    dir : Path to the directory
    transforms : Transforms to apply to the data
    batch_size : Batch size to use
    shuffle : Whether to shuffle the data defaults to False
    num_workers : Number of workers to use defaults to os.cpu_count()
    return_class_names : Whether to return the class names if True return dataloader and class names else return dataloader

  Returns:
    dataloader : DataLoader object
  """

  dataset = ImageFolder(root=dir,transform=transforms)
  class_names = dataset.classes
  dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
  if return_class_names:
    return dataloader,class_names
  return dataloader
