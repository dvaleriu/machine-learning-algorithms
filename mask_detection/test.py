import numpy as np
import torch
from datetime import datetime
import os
from Dataloader.dataloader import BaselineDataloader
from model.SimpleCnn import CNN
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    #experimentul pentru care vreau sa testez performanta modelului
    exp_name = "baseline2023-03-15-12_56_13"
    exp_path = os.path.join("experimente", exp_name)
    
    #cream folder pentru a a pendui rezultatele testelor
    results_path = os.path.join(exp_path, "test_results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    #pt reproductibilitate/deterministic
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used{device}")

    test_dataset = BaselineDataloader("Data/preprocesat", split = "split.json", phase = "test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)

    model = CNN()
    model = model.to(device)
    
    #incarcare in model ponderile cu eroarea cea mai mica
    model.load_state_dict(torch.load(os.path.join(exp_path, 'Epoch36, error0.019')))
    model.eval() #pt inferenta

    #pt evaluare de metrici 
    y_true = []
    y_pred = []

    #pt failuri
    bad_items = []
    classes_mapping = {0:'nomask', 1:'mascaprost', 2:'masca'}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            #forward pass
            output = model(images)


            #outputul e un torch tensor cu shapeul (1,3)
            #transform din torchtensor in valoarea propriuzisa
            #print('d')
            pred = torch.max(output,1)[1].data.squeeze().item()   
            
            gr = labels.item()

            y_true.append(gr)
            y_pred.append(pred)

            if pred != gr:
                bad_items.append((images.squeeze().cpu().detach().numpy(), classes_mapping[gr], classes_mapping[pred]))

    #compute metrics
    scores = f1_score(y_true, y_pred, average = None)
    print(scores)


    #matricea de confuzie
    conf_matrix = confusion_matrix(y_true=y_true, y_pred = y_pred)
    fig, ax = plt.subplots(figsize = (7.5, 7.5))
    ax.matshow(conf_matrix, cmap = plt.cm.Blues, alpha = 0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x = j, y=i, s = conf_matrix[i,j], va = 'center', ha = 'center', size = 'xx-large')


    plt.xlabel('predictions', fontsize = 18)
    plt.ylabel('gr', fontsize = 18)
    plt.title('matrice de confuzie', fontsize = 18)
    plt.savefig(os.path.join(results_path, "confusionmatrix.png"))
    plt.clf()


    #salvare cazuri gresite
    for i, data in enumerate(bad_items):
        img, gr, pred = data

        plt.imshow(img, cmap = 'gray')
        plt.savefig(os.path.join(results_path, f"Image{i}_{gr}_{pred}"))
