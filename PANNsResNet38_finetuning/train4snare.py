from pickle import FALSE, TRUE
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

import snaredataset
from snaredataset import SnareDataset
from model4snare import my_ResNet38

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def predict(model, input, target, class_mapping, batch_index):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[batch_index].argmax(0)
        predicted = class_mapping[predicted_index]
        target_index = target[batch_index]
        expected = class_mapping[target_index]
    return predicted, expected

def train_single_epoch(model, data_loader,val_loader ,loss_fn, optimiser, device, epoch):
    if epoch != 0:
        model.train()
        for input, target in data_loader:
            input = input.to(device)
            target = target.to(device)

            # calculate loss
            prediction = model(input)
            loss = loss_fn(prediction, target)

            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f"train loss: {loss.item()}")
    else:
        print("train loss: not train")

    correct = 0
    total = 0
    model.eval()
    for input, target in val_loader:
        input, target = input.to(device), target.to(device)

        prediction = model(input)
        loss = loss_fn(prediction, target)
        
        for batch_index in range(len(input)): # 128(BATCHSIZE) / 4(CLASS num) = 52
            total += 1
            predicted, expected = predict(model, input, target, class_mapping, batch_index)
            if predicted == expected:
                correct += 1

    print(f"val loss: {loss.item()}")
    print(f"val correct rate : {correct} / {total} = {correct/total}")

def test(model, test_loader, loss_fn, optimiser, device):
    correct = 0
    total = 0
    for input, target in test_loader:
        input, target = input.to(device), target.to(device)

        for batch_index in range(len(input)): # 128(BATCHSIZE) / 4(CLASS num) = 52
            total += 1
            predicted, expected = predict(model, input, target, class_mapping, batch_index)
            if predicted == expected:
                correct += 1
    print(f"test correct rate : {correct} / {total} = {correct/total}")


def train(model, train_loader, val_loader, test_loader, loss_fn, optimiser, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_single_epoch(model, train_loader,val_loader ,loss_fn, optimiser, device, epoch)
        print("---------------------------")
    print("Finished training")
    test(model, test_loader, loss_fn, optimiser, device)


if __name__ == "__main__":
    SAMPLE_RATE = 22050
    SAMPLE_NUM = 22050

    BATCH_SIZE = 128
    EPOCHS = 30
    LEARNING_RATE = 0.001

    WINDOW_SIZE = 1024
    HOP_LENGTH = 320
    MEL_BINS = 64
    FMIN = 20
    FMAX = SAMPLE_RATE/2
    CLASS_DIM = 4 #"Strike", "Rim", "Cross Stick", "Buzz"
    FREEZE_BASE = FALSE

    class_mapping = ["Strike", "Rim", "Cross Stick", "Buzz"]

    rootpath = "../data/MDLib2.2/MDLib2.2/Sorted/Snare"
    audio_path = snaredataset.get_datapath_list(rootpath)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device: {device}")

    sd = SnareDataset(audio_path,
                      SAMPLE_RATE,
                      SAMPLE_NUM,
                      device)

    sd_size = len(sd)
    train_size = int(sd_size * 0.6)
    val_size = int(sd_size * 0.2)
    test_size = sd_size - train_size - val_size

    train_dataset,val_dataset, test_dataset = torch.utils.data.random_split(sd, [train_size, val_size, test_size])
    
    train_dataloader = create_data_loader(train_dataset, BATCH_SIZE)
    val_dataloader = create_data_loader(val_dataset, BATCH_SIZE)
    test_dataloader = create_data_loader(test_dataset, BATCH_SIZE)

    model = my_ResNet38(SAMPLE_RATE,
                    WINDOW_SIZE,
                    HOP_LENGTH,
                    MEL_BINS,
                    FMIN,
                    FMAX,
                    CLASS_DIM,
                    FREEZE_BASE)

    PRETRAINED_CHECKPOINT_PATH = '../data/model/ResNet38_mAP=0.434.pth' #If you want to train without pretraining, comment out this line
    model.load_from_pretrain(PRETRAINED_CHECKPOINT_PATH)
    model = torch.nn.DataParallel(model)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimiser, device, EPOCHS)

    #torch.save(model.state_dict(), "ResNet38forSnareRecog.pth")