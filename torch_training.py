import torch
from tqdm.auto import tqdm
from typing import Dict,List,Tuple
from pathlib import Path

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim,
               loss_fn: torch.nn.Module,
               device: torch.device = "cpu") -> Tuple[float,float]:

  """
  Performs a single training step or an epoch
  Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      device: A target device to compute on (e.g. "cuda" or "cpu").
  Returns:
      A tuple of training loss and training accuracy metrics.
      In the form (train_loss, train_accuracy). For example:
  """
  model.to(device)
  model.train()

  train_loss,correct,total = 0,0,0

  for X,y in (dataloader):
      X,y = X.to(device),y.to(device)

      preds = model(X)
      loss = loss_fn(preds,y)
      train_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      y_pred_class = torch.argmax(torch.softmax(preds,dim=1),dim=1)
      correct += (y_pred_class == y).sum().item()
      total += len(y)

  train_loss = train_loss / len(dataloader)
  train_acc = correct / total

  return train_loss,train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = "cpu") -> Tuple[float,float]:

  """
  Performs a single evaluation or test step
  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    device: A target device to compute on (e.g. "cuda" or "cpu").
  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:
  """

  model.to(device)
  model.eval()

  with torch.inference_mode():

    test_loss, correct, total = 0, 0, 0
    for X,y in (dataloader):
      X,y = X.to(device),y.to(device)

      preds = model(X)
      loss = loss_fn(preds,y)
      test_loss += loss.item()

      y_pred_class = torch.argmax(torch.softmax(preds,dim=1),dim=1)
      correct += (y_pred_class == y).sum().item()
      total += len(y)

    test_loss = test_loss / len(dataloader)
    test_acc = correct / total

    return test_loss,test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device = "cpu") -> Tuple[Dict[str,List[float]],Dict[str,List[float]]]:

  """
  Peforms the training loop for the specified number of epochs
  Args:
    model: A PyTorch model to be trained.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to minimize.
    epochs: An integer indicating how many epochs to train for.
    device: "cuda" or "cpu"
  """
  model.to(device)
  results = {"train_loss":[],"train_acc":[],
             "test_loss":[],"test_acc":[]}

  curr_max_test_acc = 0
  checkpoint_path = ""

  for epoch in tqdm(range(epochs),colour="green",smoothing=0.5):

    train_loss,train_acc = train_step(model,train_dataloader,optimizer,loss_fn,device)
    test_loss,test_acc = test_step(model,test_dataloader,loss_fn,device)


    print(f"Epoch: {epoch} | Train Loss: {train_loss} | Train Acc: {train_acc} | Test Loss: {test_loss} | Test Acc: {test_acc}")
    curr_max_test_acc,checkpoint_path = model_checkpoint(model,results,test_acc,curr_max_test_acc,checkpoint_path,epoch)
    print()

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results,checkpoint_path

def model_checkpoint(
    model: torch.nn.Module,
    results: Dict[str, List[float]],
    test_acc: float,
    curr_max_test_acc: float,
    checkpoint_path: str,
    epoch: int) -> float:
    """
    Save the model checkpoint if the test accuracy improves.
    """
    Path("model-checkpoints").mkdir(parents=True, exist_ok=True)

    # If the test accuracy is better than the current max, save the checkpoint
    if test_acc > curr_max_test_acc:
        curr_max_test_acc = test_acc  # Update the current max test accuracy
        path = f"model-checkpoints/checkpoint-{epoch}.pth"
        torch.save(model.state_dict(),path)
        print(f"Saved checkpoint to model-checkpoints/checkpoint-{epoch}.pth")
        return curr_max_test_acc,path
    else:
      return curr_max_test_acc,checkpoint_path
