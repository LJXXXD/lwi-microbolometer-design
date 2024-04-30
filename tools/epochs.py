import os
import time

import numpy as np


def train_epoch(model, device, dataloader, criterions, optimizer):
    # Support for nultiple criterion functions - MSE for training and others for error display.
    train_loss = [0 for criterion in criterions]
    model.train()
    
    for samples, labels in dataloader:
        samples, labels = samples.to(device), labels.to(device)
        
        optimizer.zero_grad()

        
        # # Record the time before attempting to acquire the lock
        # start_time = time.time()
        

        output = model(samples)



        # # Record the time after the lock is acquired
        # end_time = time.time()
        # # Calculate and print the wait time
        # run_time = end_time - start_time
        # print(f"Worker process {os.getpid()} took {run_time:.10f} seconds to run x val.")
        
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



def test_epoch(model, device, dataloader, criteria):
    test_loss = [0 for criterion in criteria]
    # test_loss = np.zeros((len(dataloader.dataset), len(criterions)))
    pred_list = []
    targ_list = []
    
    model.eval()
    
    for batch_ind, batch in enumerate(dataloader):
        samples, labels = batch[0], batch[1]
        samples, labels = samples.to(device), labels.to(device)
        
        output = model(samples)

        print("test output", output)
        print("test target", labels)
        print()
        
        pred_list.append(output.detach().numpy())
        targ_list.append(labels.detach().numpy())
        
        for cri_ind, criterion in enumerate(criteria):
            loss = criterion(output, labels)
            test_loss[cri_ind] += loss.item() * samples.size(0)
            # test_loss[batch_ind, cri_ind] = loss

    return np.asarray(test_loss), np.concatenate(pred_list), np.concatenate(targ_list)