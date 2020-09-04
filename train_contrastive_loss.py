import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import time
import random
import os
import numpy as np
import torch
import configure as c
import pandas as pd
from DB_wav_reader import read_feats_structure
from SR_Dataset import read_MFB, TruncatedInputfromMFB, ToTensorInput, ToTensorDevInput, DvectorDataset, collate_fn_feat_padded
from model.model import background_resnet
from contrastive_loss import GE2ELoss

class contrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, DB, loader, spk_to_idx, spk_list, M, transform=None):
        self.DB = DB
        self.transform = transform
        self.loader = loader
        self.spk_to_idx = spk_to_idx
        self.spk_list = spk_list
        self.len = len(spk_list)
        self.M = M

    def __getitem__(self, index):
        # Index refer to a specific user: we need to load all the instances of that user
        # NOTE: the actual number of input samples is (batch_size * n_utterances_per_speaker)!!

        curr_spk = self.spk_list[index]
        idx_to_retrive = self.DB['speaker_id'] == curr_spk
        feat_paths = self.DB['filename'][idx_to_retrive]

        feat_paths = random.sample(list(feat_paths), self.M)

        tot_features = []
        tot_labels = []
        for feat_path in feat_paths:
            feature, label = self.loader(feat_path)
            label = self.spk_to_idx[label]
            label = torch.Tensor([label]).long()
            if self.transform:
                feature = self.transform(feature)

            tot_features.append(feature)
            tot_labels.append(label)

        return torch.cat(tot_features), torch.cat(tot_labels)
    
    def __len__(self):
        return self.len


def my_collate(batch):
    data = [item[0].unsqueeze(1) for item in batch]
    target = [item[1] for item in batch]

    return torch.cat(data, axis=0), torch.cat(target, axis=0)


def load_dataset(M):
    train_DB = read_feats_structure(c.TRAIN_FEAT_DIR)
    print(f'\nTraining set {len(train_DB)}')

    file_loader = read_MFB # numpy array:(n_frames, n_dims)
     
    transform = transforms.Compose([
        TruncatedInputfromMFB(), # numpy array:(1, n_frames, n_dims)
        ToTensorInput() # torch tensor:(1, n_dims, n_frames)
    ])
    transform_T = ToTensorDevInput()
   
    
    speaker_list = sorted(set(train_DB['speaker_id'])) # len(speaker_list) == n_speakers
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    
    train_dataset = contrastiveDataset(DB=train_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx, spk_list=speaker_list, M=M)

    n_classes = len(speaker_list) # How many speakers? 240
    print('\nNumber of classes (speakers):\n{}\n'.format(n_classes))
    return train_dataset, n_classes


def main():
    
    # GPU parmas
    idx_cuda = -1
    if idx_cuda < 0:
        device = torch.device('cpu')
    else:    
        device = torch.device(f'cuda:{idx_cuda}')
    

    # Set hyperparameters
    embedding_size = 128 # origial 128
    start = 1 # Start epoch
    n_epochs = 10 # How many epochs?
    end = start + n_epochs # Last epoch
    
    lr = 1e-1 # Initial learning rate
    wd = 1e-4 # Weight decay (L2 penalty)
    optimizer_type = 'adam' # ex) sgd, adam, adagrad
    
    #batch_size = 4 # Batch size for training
    M = 2 # number of utterances per speaker
    N = 8  # Number of speaker

    # The effective batch size is M*N

    print(f'Number of utternaces per speaker: {M}')
    print(f'Number of speakers: {N}')
    print(f'** Total batch size: {M*N} **')

    use_shuffle = True # Shuffle for training or not
    save_interval = 5
    
    # Load dataset
    train_dataset, n_classes = load_dataset(M)
    
    # print the experiment configuration

    
    log_dir = 'model_saved' # where to save checkpoints
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # instantiate model and initialize weights
    model = background_resnet(embedding_size=embedding_size, num_classes=n_classes, backbone='resnet18')

    # Load the wights trained for identification
    #model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoint_45.pth')))

    # remove the last layers
    
    model.to(device)
    
    # define loss function (criterion), optimizer and scheduler
    criterion = GE2ELoss(device)
    optimizer = create_optimizer(optimizer_type, model, lr, wd)


    # Define dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=N,
                                                       shuffle=use_shuffle,
                                                collate_fn=my_collate)

                               
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []

    
    for epoch in range(start, end):
    
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, device, epoch, N, M)

        # calculate average loss over an epoch
        avg_train_losses.append(train_loss)
        
        # do checkpointing
        if epoch % save_interval == 0 or epoch == end-1:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       '{}/checkpoint_{}.pth'.format(log_dir, epoch))


def train(train_loader, model, criterion, optimizer, device, epoch, N, M):
    batch_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    
    n_correct, n_total = 0, 0
    log_interval = 84
    # switch to train mode
    model.train()
    
    end = time.time()
    # pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data) in enumerate(train_loader):
        inputs, targets = data  # target size:(batch size,1), input size:(batch size, 1, dim, win)
        targets = targets.view(-1) # target size:(batch size)
        current_sample = inputs.size(0)  # batch size

        # Zero out gradinets
        optimizer.zero_grad()
       
    
        inputs = inputs.to(device)
        targets = targets.to(device)
    

        spk_embedding, output = model(inputs) # out size:(batch size, #classes), for softmax
        
        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
        n_total += current_sample
        train_acc_temp = 100. * n_correct / n_total
        train_acc.update(train_acc_temp, current_sample)
        
        spk_embedding_reshape = spk_embedding.reshape(N, M, -1)

        loss = criterion(spk_embedding_reshape)

        losses.update(loss.item(), current_sample)
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            print(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.avg:.4f}\t'
                    'Acc {train_acc.avg:.4f}'.format(
                     epoch, batch_idx * len(inputs), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), 
                     batch_time=batch_time, loss=losses, train_acc=train_acc))
    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_optimizer(optimizer, model, new_lr, wd):
    # setup optimizer
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0,
                              weight_decay=wd)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=wd)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  weight_decay=wd)
    return optimizer

if __name__ == '__main__':
    main()