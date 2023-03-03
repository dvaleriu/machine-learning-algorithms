import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random


def plot_data(features, centroid_points):
    plt.scatter(features[:,0], features[:,1])
    plt.scatter(centroid_points[:,0], centroid_points[:,1], color = 'r')
    plt.title("centroiziii")
    plt.xlabel("venit anual")
    plt.ylabel("cheltuieli")
    plt.show()

def euclid_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def assign_samples_to_centroids(features, centroid_points):
    clusters = []
    #calculam mai intai distanta de la fiecare centroid la fiecare sample
    #astfel incat sa putem asigna punctele la clusterul bun
    for samples in features:
        dist = [euclid_distance(samples,x) for x in centroid_points]
        idx_min = np.argmin(dist)
        clusters.append(idx_min)
    return np.array(clusters)


def caclulate_new_centroids(clusters ,k):
    for idx in range(0,k):
        #extrag indexii centroidului
        points_idx = np.argwhere(clusters == idx)
        #extrac punctele
        points = features[points_idx]
        #pt axele cu dim 1:
        points = points.squeeze()
        #media pe Ox si Oy:
        x = np.mean(points[:,0])
        y = np.mean(points[:,1])
        #actualizare
        centroid_points[idx][0] = x
        centroid_points[idx][1] = y
    return centroid_points


if __name__ == "__main__" :
    #citire fisier
    dataset = pd.read_csv("Mall_Customers.csv", index_col = 'CustomerID')

    #analiza datelor:
    dataset.head()
    dataset.info()
    print(dataset.shape)

    #transform dataframeul intr un numpy array si extrag doar ultimele 2 caracteristici
    features = np.array(dataset)[:,2:]
    print(features.shape)

    """
    plt.scatter(features[:,0], features[:,1])
    plt.title("Vizualizare")
    plt.xlabel("Venit anual")
    plt.ylabel("Cheltuieli")
    plt.show()
    """
    k = 5 #clustere, 
    init_centroids = random.sample(range(0,features.shape[0]),k)
    print(f"centroizii alesi sunt {init_centroids}")

    #extrag puncte + adaugare in nparray
    centroid_points =  np.array([features[init_centroids[x]] for x in range(0, k)]) #5x2 shape: 5 clustere a cate 2 dim fiecare
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    #print("punctele corespunzatoare centroizi: ")
    #plot_data(features, centroid_points)

    #pt fiecare element: 0- apartine grupului 0, 1 pt 1 samd
    clusters = assign_samples_to_centroids(features, centroid_points)

    number_of_iterations = 10
    for idx in range(number_of_iterations):
    # Calculam noile pozitii ale centroizilor
        centroid_points = caclulate_new_centroids(clusters, k)
   
    # Asignam fiecare sample cate unui centroid, considerand lista noua
        clusters = assign_samples_to_centroids(features, centroid_points)
     # Vizualizam cum arata acest proces de la o iteratia la alta
        plot_data(features, centroid_points)
   
