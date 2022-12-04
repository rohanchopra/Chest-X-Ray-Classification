import time

#from torchmetrics import F1Score
import torch
import torch.autograd.profiler as tprofiler
import copy
import numpy as np


def train_model(device, model, dataloaders, optimizer, scheduler, criterion, 
                num_epochs, num_classes, is_inception=False, profiler=False):
    start = time.time()

    val_acc_history = []
    val_loss_history = []
    val_outputs_history=[]
    val_targets_history=[]
    train_acc_history = []
    train_loss_history = []
    train_outputs_history=[]
    train_targets_history=[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    prof = None
    
    #f1 = F1Score(num_classes=num_classes)
    
    total_steps = len(dataloaders['train'])

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode.
            else:
                model.eval()   # Set model to evaluation mode.

            running_loss = 0.0
            running_corrects = 0

            i = 0
            
            targets_batch = []
            outputs_batch = []
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                i+=1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Make the parameter gradients zero.
                # optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None
                    # print  requires grad to show its true

                # Forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model loss and outputs
                    # Special case for inception- in training has an auxiliary output
                    # In training calculate the loss by summing the final output and the auxiliary output
                    # In testing use only the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        if profiler:
                            with tprofiler.profile(with_stack=True, profile_memory=True) as prof:
                                outputs, aux_outputs = model(inputs)
                        else:
                            outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        if profiler:
                              with tprofiler.profile(with_stack=True, profile_memory=True) as prof:
                                outputs = model(inputs)
                        else:
                              outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    # statistics
                    curr_loss = loss.item()
                    running_loss += curr_loss * inputs.size(0)
                    
                    curr_acc = torch.sum(preds == labels.data)
                    running_corrects += curr_acc
                    
                    outputs_batch.append(preds)
                    targets_batch.append(labels.data)

                if (i) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{total_steps}], Loss: {curr_loss:.4f}, Accuracy: {curr_acc/preds.shape[0]:.2f}%')

            with torch.no_grad():
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # Deep copy the best performing model.
                # Add different metrics to history.
                # Use F! score here----
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    outputs_batch = [i.cpu().detach().numpy() for i in outputs_batch]
                    targets_batch = [i.cpu().detach().numpy() for i in targets_batch]
                    outputs_batch = np.concatenate(outputs_batch)
                    targets_batch = np.concatenate(targets_batch)
                    val_outputs_history.append(outputs_batch)
                    val_targets_history.append(targets_batch)
                    #f1 = f1_score(targets, outputs, average='macro')
                    #val_f1_history.append(f1)
                    
                    val_acc_history.append(epoch_acc.item())
                    val_loss_history.append(epoch_loss)
                elif phase == 'train':
                    
                    outputs_batch = [i.cpu().detach().numpy() for i in outputs_batch]
                    targets_batch = [i.cpu().detach().numpy() for i in targets_batch]
                    outputs_batch = np.concatenate(outputs_batch)
                    targets_batch = np.concatenate(targets_batch)
                    train_outputs_history.append(outputs_batch)
                    train_targets_history.append(targets_batch)
                    #f1 = f1_score(targets, outputs, average='macro')
                    #train_f1_history.append(f1)
                    
                    
                    train_acc_history.append(epoch_acc.item())
                    train_loss_history.append(epoch_loss)

        scheduler.step()
        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights to return
    model.load_state_dict(best_model_wts)
    return model, prof, {'acc': val_acc_history, 'loss': val_loss_history, 'targets': val_targets_history, 'outputs': val_outputs_history}, \
           {'acc': train_acc_history, 'loss': train_loss_history, 'targets': train_targets_history, 'outputs': train_outputs_history}