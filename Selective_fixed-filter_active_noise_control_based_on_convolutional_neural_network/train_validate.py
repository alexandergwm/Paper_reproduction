from os import write
import os
from sklearn.utils import shuffle
import torch
from torch import nn
from torch.utils.data import DataLoader
from MyDataLoader import MyNoiseDataset
from ONED_CNN import OneD_CNN, OneD_CNN_with_Res
from Bcolors import bcolors
from TWOD_CNN import Modified_ShufflenetV2

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.00005

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    train_loss = 0
    train_acc = 0
    model.train()
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss       = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # save the loss and accuracy
        train_loss += loss.item()
        _, pred = prediction.max(1)
        num_correct = (pred == target).sum().item()
        acc         = num_correct/input.shape[0]
        train_acc  += acc
    
    print(f'loss: {train_loss/len(data_loader)}')
    print(f'Accuracy: {train_acc / len(data_loader)}')
    return train_acc / len(data_loader)

def validate_single_epoch(model, eva_data_loader, loss_fn, device):
    eval_loss = 0
    eval_acc = 0
    model.eval()
    i         = 0
    for input, target in eva_data_loader:
        input, target = input.to(device), target.to(device)
        i            += 1
        # Calculating the loss value
        prediction = model(input)
        loss       = loss_fn(prediction, target)

        # save the validating loss and accuracy
        eval_loss += loss.item()
        _, pred = prediction.max(1)
        num_correct = (pred == target).sum().item()
        acc         = num_correct / input.shape[0]
        eval_acc   += acc
        # Break the foor loop 
        if i == 40:
            break 

    print(f"Validat loss : {eval_loss/i}" + f" Validat accuracy : {eval_acc/i}") 
    return eval_acc/i


def train(model, data_loader, eva_data_loader, loss_fn, optimiser, device, epochs, MODEL_PTH=None):
    acc_max = 0
    acc_train_max = 0
    stop_epoch = 5
    for i in range(epochs):
        print(f"Epoch {i+1}")
        acc_train = train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        acc_validate = validate_single_epoch(model, eva_data_loader, loss_fn, device)
        if acc_validate > acc_max:
            acc_train_max, acc_max = acc_train, acc_validate
            torch.save(model.state_dict(), MODEL_PTH)
            print(bcolors.OKGREEN + "Trained feed forward net save at" + MODEL_PTH + bcolors.ENDC)
            no_improve_counts = 0
        else:
            no_improve_counts += 1
        print("-----------------------------------")

        # decide early stop or not
        if no_improve_counts >= stop_epoch:
            print(f"Stopping early as validation accuracy did not improve for {stop_epoch} consecutive epochs.")
            break

    print("Finished training")
    return acc_train_max, acc_max


def Train_validate_predefined_1DCNN(TRAIN_DATASET_FILE, VALIDATION_DATASET_FILE, MODEL_PTH):
    """
    * Training and validating the pre-defined 1D-CNN
    """
    File_sheet = 'Index.csv'

    train_data = MyNoiseDataset(TRAIN_DATASET_FILE, File_sheet)
    valid_data = MyNoiseDataset(VALIDATION_DATASET_FILE, File_sheet)

    train_dataloader = create_data_loader(train_data, BATCH_SIZE)
    valid_dataloader = create_data_loader(valid_data, int(BATCH_SIZE/10))

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    feed_forward_net_1D = OneD_CNN().to(device)
    # initialize loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net_1D.parameters(), lr=LEARNING_RATE)
    # train model
    train(feed_forward_net_1D, train_dataloader, valid_dataloader, loss_fn, optimiser, device, EPOCHS, MODEL_PTH=MODEL_PTH)
    
    # save model
    torch.save(feed_forward_net_1D.state_dict(), "feedforwardnet_1D.pth")
    print("Trained feed forward net saved at feedforwardnet_1D.pth")


def Train_validate_predefined_2DCNN(TRAIN_DATASET_FILE, VALIDATION_DATASET_FILE, MODEL_PTH):
    """
    * Training and validating the pre-defined 2D-CNN
    """
    File_sheet = 'Index.csv'

    train_data = MyNoiseDataset(TRAIN_DATASET_FILE, File_sheet,use_stft=True)
    valid_data = MyNoiseDataset(VALIDATION_DATASET_FILE, File_sheet,use_stft=True)

    train_dataloader = create_data_loader(train_data, BATCH_SIZE)
    valid_dataloader = create_data_loader(valid_data, int(BATCH_SIZE/10))

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    feed_forward_net_2D =  Modified_ShufflenetV2(15).to(device)
    print(feed_forward_net_2D) 

    # initialize loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net_2D.parameters(), lr=LEARNING_RATE)

    # train model
    train(feed_forward_net_2D, train_dataloader, valid_dataloader, loss_fn, optimiser, device, EPOCHS, MODEL_PTH=MODEL_PTH)
    
    # save model
    torch.save(feed_forward_net_2D.state_dict(), "feedforwardnet_2D.pth")
    print("Trained feed forward net saved at feedforwardnet_2D.pth")


if __name__ == "__main__":
    File_sheet = 'Index.csv'
    TRAIN_DATASET_FILE = "D:\Coding\Gavin\Selective_ANC_CNN\Training_data"
    VALIDATION_DATASET_FILE = 'D:\Coding\Gavin\Selective_ANC_CNN\Validating_data'
    MODEL_PTH = r"/Coding/Gavin/Selective_ANC_CNN/model.pth"

    method = "2DCNN"
    if method == "1DCNN":
        MODEL_PTH = 'feedforwardnet_1D.pth'
        Train_validate_predefined_1DCNN(TRAIN_DATASET_FILE, VALIDATION_DATASET_FILE, MODEL_PTH)

    elif method == "2DCNN":
        MODEL_PTH = 'feedforwardnet_2D.pth'
        Train_validate_predefined_2DCNN(TRAIN_DATASET_FILE, VALIDATION_DATASET_FILE, MODEL_PTH)

    
