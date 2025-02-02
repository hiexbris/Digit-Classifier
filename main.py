import torchvision.datasets as datasets
import numpy as np


def softmax(Z3):
    exp_Z3 = np.exp(Z3 - np.max(Z3, axis=0))  
    return exp_Z3 / np.sum(exp_Z3, axis=0, keepdims=True)


def load_data():
    mnist_train = datasets.MNIST(root="./data", train=True, download=True)
    x_train = np.array(mnist_train.data)  
    y_train = np.array(mnist_train.targets) 
    x_train = x_train.reshape(60000, 784)[:, :728]  
    x_train = x_train.T  
    y_train = y_train.reshape(1, 60000)

    return x_train, y_train


x_train, y_train = load_data()

def check_image(x_train, y_train, num): 

    first_image = np.zeros(784)  
    first_image[:728] = x_train[:, num]  
    first_image = first_image.reshape(28, 28) 
    print(y_train[0, num])
    from PIL import Image
    image = Image.fromarray(first_image.astype(np.uint8))
    image.save("Image.png")


# check_image(x_train, y_train, num=1000)
y_train = np.eye(10)[y_train].squeeze().T  

print(x_train.shape)

W1 = np.random.randn(128, 728) * np.sqrt(2 / 728)
W2 = np.random.randn(128, 128) * np.sqrt(2 / 128)
W3 = np.random.randn(10, 128) * np.sqrt(2 / 128)
B1 = np.zeros((128, 1))
B2 = np.zeros((128, 1))
B3 = np.zeros((10, 1))

for i in range(0, x_train.shape[1], 32):  
    S0 = x_train[:, i:i + 32]  
    Y = y_train[:, i:i + 32]

    Z1 = np.dot(W1, S0) + B1
    S1 = np.maximum(0, Z1)

    Z2 = np.dot(W2, S1) + B2
    S2 = np.maximum(0, Z2)

    Z3 = np.dot(W3, S2) + B3
    S3 = softmax(Z3)

    E3 = Y * np.log(S3) + (1 - Y) * np.log(1 - S3)

    Z2 = np.where(Z2 > 0, 1, 0)
    E2 = np.dot(W3.T, E3) * Z2

    Z1 = np.where(Z1 > 0, 1, 0)
    E1 = np.dot(W2.T, E2) * Z1

    F3 = 