from tqdm import tqdm
import torch
import config
import numpy as np
from utils.model_utils import save_model_checkpoint


def train_fn(model, data_loader, optimizer, loss_fn):
    model.train()
    fin_loss = 0

    target_hr_list = []
    predicted_hr_list = []
    tk_iterator = tqdm(data_loader, total=len(data_loader))

    for batch in tk_iterator:
        if config.OPTIMIZER == 'noam':
            optimizer.optimizer.zero_grad()
        else:
            optimizer.zero_grad()
        inputs = torch.stack(tuple(data['data'] for data in batch))
        targets = torch.stack(tuple(data['target'] for data in batch))

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets.view(outputs.shape))
            loss.backward()
            optimizer.step()
            for i in range(len(targets)):
                predicted_hr_list.append(outputs[i].item())
                target_hr_list.append(targets[i].item())
            fin_loss += loss.item()

        #for data in batch:
            # an item of the data is available as a dictionary
        #    for (key, value) in data.items():
        #        data[key] = value.to(config.DEVICE)
        #    if config.OPTIMIZER == 'noam':
        #        optimizer.optimizer.zero_grad()
        #    else:
        #        optimizer.zero_grad()
        #
        #    with torch.set_grad_enabled(True):
        #        outputs = model(data["data"])
        #
        #        loss = loss_fn(outputs, data["target"].view(outputs.shape))
        #        predicted_hr_list.append(outputs.mean().item())
        #
        #        loss.backward()
        #        optimizer.step()
        #
        #    target_hr_list.append(data["target"].mean().item())
        #
        #    fin_loss += loss.item()

    return target_hr_list, predicted_hr_list, fin_loss / (len(data_loader))


def eval_fn(model, data_loader, loss_fn):
    model.eval()
    fin_loss = 0
    target_hr_list = []
    predicted_hr_list = []

    with torch.no_grad():

        tk_iterator = tqdm(data_loader, total=len(data_loader))
        for batch in tk_iterator:
            inputs = torch.stack(tuple(data['data'].to(config.DEVICE) for data in batch))
            targets = torch.stack(tuple(data['target'].to(config.DEVICE) for data in batch))

            outputs = model(inputs)
            loss = loss_fn(outputs, targets.view(outputs.shape))

            for i in range(len(targets)):
                predicted_hr_list.append(outputs[i].item())
                target_hr_list.append(targets[i].item())
            fin_loss += loss.item()

        return target_hr_list, predicted_hr_list, fin_loss / (len(data_loader))
