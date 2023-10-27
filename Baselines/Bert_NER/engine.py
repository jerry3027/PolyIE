import torch
from tqdm import tqdm
import torch.nn.functional as F
import config

from transformers import BertTokenizer


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss, _, _, _ = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        loss, _, _, _ = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)