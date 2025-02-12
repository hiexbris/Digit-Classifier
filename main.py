import torchvision.datasets as datasets
import numpy as np   


def softmax(Z3):
    exp_Z3 = np.exp(Z3 - np.max(Z3, axis=0))  
    return exp_Z3 / (np.sum(exp_Z3, axis=0, keepdims=True) + 1e-8)


def load_data():
    mnist_train = datasets.MNIST(root="./data", train=True, download=True)
    x_train = np.array(mnist_train.data)  
    y_train = np.array(mnist_train.targets) 
    x_train = x_train.reshape(60000, 784)
    x_train = x_train.T  
    y_train = y_train.reshape(1, 60000)

    return x_train, y_train


x_train, y_train = load_data()


def check_image(x_train, y_train, num): 

    first_image = np.zeros(784)  
    first_image[:784] = x_train[:, num]  
    first_image = first_image.reshape(28, 28) 
    print(y_train[0, num])
    from PIL import Image
    image = Image.fromarray(first_image.astype(np.uint8))
    image.save("Image.png")


# check_image(x_train, y_train, num=0)
x_train = x_train.astype(np.float32) / 255.0
y_train = np.eye(10)[y_train].squeeze().T  


W1 = np.random.randn(128, 784) * np.sqrt(2 / 784)
W2 = np.random.randn(128, 128) * np.sqrt(2 / 128)
W3 = np.random.randn(10, 128) * np.sqrt(2 / 128)
B1 = np.zeros((128, 1))
B2 = np.zeros((128, 1))
B3 = np.zeros((10, 1))

alpha = 0.01
batch = 1

for epoch in range(5):
    for i in range(0, x_train.shape[1], batch):  
        print(epoch, i)
        S0 = x_train[:, i:i + batch]  
        Y = y_train[:, i:i + batch]

        Z1 = np.dot(W1, S0) + B1
        S1 = np.maximum(0, Z1)

        Z2 = np.dot(W2, S1) + B2
        S2 = np.maximum(0, Z2)

        Z3 = np.dot(W3, S2) + B3
        S3 = softmax(Z3)
        
        # E3 = Y * np.log(S3 + epsilon) + (1 - Y) * np.log(1 - S3 + epsilon)
        loss = -np.sum(Y*np.log(S3 + 1e-8)) / batch
        E3 = S3 - Y 

        Z2 = np.where(Z2 > 0, 1, 0)
        E2 = np.dot(W3.T, E3) * Z2

        Z1 = np.where(Z1 > 0, 1, 0)
        E1 = np.dot(W2.T, E2) * Z1

        F3 = np.dot(E3, S2.T) / batch
        F2 = np.dot(E2, S1.T) / batch
        F1 = np.dot(E1, S0.T) / batch

        W3 = W3 - (alpha)*F3
        W2 = W2 - (alpha)*F2
        W1 = W1 - (alpha)*F1

        E3 = np.mean(E3, axis=1).reshape(10, 1)
        E2 = np.mean(E2, axis=1).reshape(128, 1)
        E1 = np.mean(E1, axis=1).reshape(128, 1)

        B3 = B3 - (alpha)*E3
        B2 = B2 - (alpha)*E2
        B1 = B1 - (alpha)*E1

        del S0, Y, Z1, S1, Z2, S2, Z3, S3, F1, F2, F3, E3, E2, E1


def load_test():

    mnist_test = datasets.MNIST(root="./data", train=False, download=True)

    x_test = np.array(mnist_test.data).reshape(10000, 784).T     
    y_test = np.array(mnist_test.targets).reshape(1, 10000)    

    return x_test, y_test


x_test, y_test = load_test()
x_test = x_test.astype(np.float32) / 255.0

Z1 = np.dot(W1, x_test) + B1
S1 = np.maximum(0, Z1)

Z2 = np.dot(W2, S1) + B2
S2 = np.maximum(0, Z2)

Z3 = np.dot(W3, S2) + B3
S3 = softmax(Z3)

predictions = np.argmax(S3, axis=0).reshape(1, 10000)

import pandas as pd

results = pd.DataFrame({
    'Predicted': predictions.flatten(),
    'Actual': y_test.flatten()
})

results.to_csv('predictions_comparison.csv', index=False)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")

import torch

weights = {
    'W1': W1,
    'W2': W2,
    'W3': W3,
    'B1': B1,
    'B2': B2,
    'B3': B3
}

torch.save(weights, 'weights.pth')