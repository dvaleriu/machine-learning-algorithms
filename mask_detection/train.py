from comet_ml import Experiment
import numpy as np
import torch
from datetime import datetime
import os
from Dataloader.dataloader import BaselineDataloader
from model.SimpleCnn import CNN
from matplotlib import pyplot as plt


if __name__ == "__main__":
    #experiment
    exp_name = "baseline"
    exp_path = os.path.join("experimente", exp_name + datetime.today().strftime('%Y-%m-%d-%H_%M_%S'))
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    experiment = Experiment(api_key="baGXqK4GkaHf5E85e1nGf12Hh",project_name="faia",workspace="dvaleriu",)

    #pt reproductibilitate/deterministic
    random_seed = 1
    torch.manual_seed(random_seed)

    #hyperparameters
    epochs = 100
    batch_size_train = 32
    learning_rate = 0.003

    #on gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used{device}")

    #dataseturi
    train_dataset = BaselineDataloader("Data/preprocesat", split = "split.json", phase = "train")
    val_dataset = BaselineDataloader("Data/preprocesat", split = "split.json", phase = "val")

    #dataloadere
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size_train, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle = False)

    #model
    model = CNN()
    model = model.to(device)

    #nr  parametri antrenabili model
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Nr total de param{model_total_params}")
    experiment.log_parameter("nr_of_model_params", model_total_params) #pt comet

    #definire loss
    criterion = torch.nn.NLLLoss()
    
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


    ########################### TRAIN #########################
    errors_train = []
    errors_validation = []

    for epoch in range(epochs):
        temporal_loss_train = []

        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            #curatare gradienti
            optimizer.zero_grad()

            #forward prop
            output = model(images)

            #compute error
            loss = criterion(output, labels)
            temporal_loss_train.append(loss.item())

            #backprop
            loss.backward()

            #update
            optimizer.step()


        model.eval()
        temporal_loss_valid = []
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            #forward pass
            output = model(images)

            #eroare
            loss = criterion(output, labels)
            temporal_loss_valid.append(loss.item())

            #eroare medie
            medium_epoch_loss_train = sum(temporal_loss_train)/len(temporal_loss_train)
            medium_epoch_loss_valid = sum(temporal_loss_valid)/len(temporal_loss_valid)

            errors_train.append(medium_epoch_loss_train)
            errors_validation.append(medium_epoch_loss_valid)

        print(f"Epoca{epoch}. Training loss{medium_epoch_loss_train},, val loss{medium_epoch_loss_valid}")

        #pt comet_ml
        experiment.log_metric("train_loss", medium_epoch_loss_train, step = epoch)
        experiment.log_metric("valid_loss", medium_epoch_loss_valid)

        #salvare model
        torch.save(model.state_dict(), os.path.join(exp_path, f"Epoch{epoch}, error{round(medium_epoch_loss_valid,3)}")) 


    plt.title("curbe antrenare")
    plt.plot(errors_train, label = "train")
    plt.plot(errors_validation, label = "valid")
    plt.xlabel("epoci")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(exp_path, "losses.png"))










