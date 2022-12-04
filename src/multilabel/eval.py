from sklearn.metrics import multilabel_confusion_matrix, f1_score
import torch
import numpy as np


def eval_model(device, model, test_loader, is_inception, num_classes=7):
    # Make the parameter gradients zero.
    for param in model.parameters():
        param.grad = None
    model.eval()
    i=0
    outputs_batch=[]
    targets_batch=[]
    running_corrects = 0
    for inputs, labels in test_loader:
        i+=1
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        # track history if only in train
        with torch.no_grad():
            # Get model loss and outputs
            # Special case for inception- in training has an auxiliary output
            # In training calculate the loss by summing the final output and the auxiliary output
            # In testing use only the final output.
            if is_inception:
                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                outputs, aux_outputs = model(inputs)
            else:
                outputs = model(inputs)

            # _, preds = torch.max(outputs, 1)
            preds = torch.clone(outputs)
            preds[preds>=0.5] = 1
            preds[preds<0.5] = 0

            # statistics
            curr_acc = torch.sum(preds == labels.data)
            running_corrects += curr_acc

            outputs_batch.append(preds)
            targets_batch.append(labels.data)
            
    outputs_batch = [i.cpu().detach().numpy() for i in outputs_batch]
    targets_batch = [i.cpu().detach().numpy() for i in targets_batch]
    outputs_batch = np.concatenate(outputs_batch)
    targets_batch = np.concatenate(targets_batch)

    test_acc = running_corrects.double() / (len(test_loader.dataset) * num_classes)
    test_f1 = f1_score(targets_batch, outputs_batch, average='macro')
    test_cm = multilabel_confusion_matrix(targets_batch, outputs_batch)
                     
    return {'acc': test_acc.item(), 'f1': test_f1, 'cm': test_cm, 'outputs': outputs_batch, 'targets': targets_batch}

