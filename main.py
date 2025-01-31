import torchvision.datasets as datasets
import numpy as np


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
y_one_hot = np.zeroes((10, len(y_train)))

y_one_hot[y_train, np.arange(len(y_train))] = 1

print(y_one_hot.shape)