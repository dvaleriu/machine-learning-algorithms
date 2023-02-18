import numpy as np
from matplotlib import pyplot as plt

def mse(y_pred, y_gr):
    return np.mean((y_pred - y_gr)**2)

 #date nelin
def generate_data(num_samples=30):
    X = np.random.rand(1000)
    y = 5*((X)**(3)) + np.random.rand(1000)
    return X,y

if __name__ == '__main__':
    features, target = generate_data(30)
    

    #learning rate
    lr = 0.003

    #nr de epoci
    epochs = 500

    #initializam ht(x)
    t0, t1, t2, t3 = 1.2, 3.4, 3.2, 2.8

    dataset = [(x,y) for x,y in zip(features, target)]

    errors = []
  
    eroare_medie_pe_epoca = []
    for epoch in range(epochs):
        errors = []
        for x,y_gr in dataset:

            #predictie 
            y_pred = t0* x ** 3 + t1 * x ** 2 + t2 * x + t3

            #calculam eroarea
            error = mse(y_pred, y_gr)
            errors.append(error)

            #calcul derivate
            t0_grad = x**3 * (y_pred - y_gr)
            t1_grad = x**2 * (y_pred - y_gr) 
            t2_grad = x * (y_pred - y_gr)
            t3_grad = y_pred - y_gr

            #update parametri
            t0 = t0 - lr * t0_grad
            t1 = t1 - lr * t1_grad
            t2 = t2 - lr * t2_grad
            t3 = t3 - lr * t3_grad

        eroare_medie_pe_epoca.append(np.mean(errors))

#plt.plot(eroare_medie_pe_epoca)
preds = t0* features ** 3 + t1 * features ** 2 + t2 * features + t3
plt.scatter(features, target )
plt.scatter(features, preds )
plt.show()           