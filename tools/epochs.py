
import numpy as np


def train_epoch(model, device, dataloader, criterions, optimizer):
    # Support for nultiple criterion functions - MSE for training and others for error display.
    train_loss = [0 for criterion in criterions]
    model.train()
    
    for samples, labels in dataloader:
        samples, labels = samples.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(samples)
        
        # Training criterion is the last in the list, so that the loss could be used directly for back prop.
        for i, criterion in enumerate(criterions):
            loss = criterion(output, labels)
            train_loss[i] += loss.item() * samples.size(0)
        
        loss.backward()
        optimizer.step()
        
    return np.asarray(train_loss)



def valid_epoch(model, device, dataloader, criterions):
    valid_loss = [0 for criterion in criterions]
    model.eval()
    
    for samples, labels in dataloader:
        samples, labels = samples.to(device), labels.to(device)
        
        output = model(samples)
        
        for i, criterion in enumerate(criterions):
            loss = criterion(output, labels)
            valid_loss[i] += loss.item() * samples.size(0)
        
    return np.asarray(valid_loss)



def test_epoch(model, device, dataloader, criterions):
    test_loss = [0 for criterion in criterions]
    pred_list = []
    targ_list = []
    
    model.eval()
    
    for samples, labels in dataloader:
        samples, labels = samples.to(device), labels.to(device)
        
        output = model(samples)
        
        pred_list.append(output.detach().numpy())
        targ_list.append(labels.detach().numpy())
        
        for i, criterion in enumerate(criterions):
            loss = criterion(output, labels)
            test_loss[i] += loss.item()
            
    return np.asarray(test_loss), pred_list, targ_list