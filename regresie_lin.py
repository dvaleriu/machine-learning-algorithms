import numpy as np
from matplotlib import pyplot as plt

def mse(y_pred, y_gr):
    return np.mean((y_pred - y_gr)**2)


def generate_data(num_samples=30):
    X = np.array(range(num_samples))
    random_noise = np.random.uniform(10, 40, size= num_samples)
    y = 3.5 * X + random_noise
    return X,y

if __name__ == '__main__':
    features, target = generate_data(30)
    

    #learning rate
    lr = 0.003

    #nr de epoci
    epochs = 500

    #initializam ht(x)
    t0,t1 = 1.2, 3.4

    dataset = [(x,y) for x,y in zip(features, target)]

    errors = []
  
    eroare_medie_pe_epoca = []
    for epoch in range(epochs):
        errors = []
        for x,y_gr in dataset:

            #predictie 
            y_pred = t1 * x + t0

            #calculam eroarea
            error = mse(y_pred, y_gr)
            errors.append(error)

            #calcul derivate
            t1_grad =(y_pred - y_gr) * x
            t0_grad = y_pred - y_gr

            #update parametri
            t0 = t0 - lr * t0_grad
            t1 = t1 - lr * t1_grad
        eroare_medie_pe_epoca.append(np.mean(errors))
plt.plot(eroare_medie_pe_epoca)
plt.show()           