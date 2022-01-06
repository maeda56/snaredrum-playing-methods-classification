import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

import snaredataset
from snaredataset import SnareDataset
from cnn import CNNNetwork

class_mapping = ["Strike", "Rim", "Cross Stick", "Buzz"]

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

audio_path = snaredataset.get_datapath_list()
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


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

def train_single_epoch(model, data_loader,test_loader ,loss_fn, optimiser, device, epoch):
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
    for input, target in test_loader:
        input, target = input.to(device), target.to(device)

        prediction = model(input)
        loss = loss_fn(prediction, target)

        
        
        for batch_index in range(len(input)): # 128(BATCHSIZE) / 4(CLASS num) = 52
            total += 1
            predicted, expected = predict(model, input, target, class_mapping, batch_index)
            if predicted == expected:
                correct += 1

    print(f"val loss: {loss.item()}")
    print(f"correct rate : {correct} / {total} = {correct/total}")


def train(model, data_loader, test_loader, loss_fn, optimiser, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_single_epoch(model, data_loader,test_loader ,loss_fn, optimiser, device, epoch)
        print("---------------------------")
    print("Finished training")

def test(model, test_loader, device):
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


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    sd = SnareDataset(audio_path,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    sd_size = len(sd)
    train_size = int(sd_size * 0.6)
    val_size = int(sd_size * 0.2)
    test_size = sd_size - train_size - val_size 

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(sd, [train_size, val_size, test_size])
    
    train_dataloader = create_data_loader(train_dataset, BATCH_SIZE)
    val_dataloader = create_data_loader(val_dataset, BATCH_SIZE)
    test_dataloader = create_data_loader(test_dataset, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)
    test(cnn, test_dataloader, device)

    # save model
    #torch.save(cnn.state_dict(), "CNNnet.pth")
    #print("Trained feed forward net saved at CNNnet.pth")