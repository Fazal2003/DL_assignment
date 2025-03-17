import numpy as np
import wandb 
import random
from layer import layer
from activation import activation
from loss import loss
from keras.datasets import fashion_mnist
# making a neural network to classify the fashion mnsit data set

run = wandb.init(
    # Set the wandb project where this run will be logged.
    project="DL Assignment",
)

epochs = 10

(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

size=len(train_images)
def one_hot_encode(labels,classes):
    one_hot=np.zeros((size,classes))
    one_hot[np.arange(labels.shape[0]),labels]=1
    return one_hot
y_true=one_hot_encode(train_labels,10)

n=3 #number of layers
layer1=layer(784,16)
layer2=layer(16,16)
layer3=layer(16,10)
activ=activation()
cost=loss()
learning_rate=0.005

train_images, test_images = train_images/255.0, test_images/255.0
train_images, test_images = train_images.reshape(-1, 784), test_images.reshape(-1, 784)

z1 = layer1.forward(train_images)
a1=activ.relu(z1)
z2=layer2.forward(a1)
a2=activ.relu(z2)
z3=layer3.forward(a2)
a3=activ.sftmx(z3)
a3=np.clip(a3,1e-7,1-1e-7)

l=cost.cross_entropy(a3,y_true)
wandb.log({"loss": l})

dL_dz3=a3-y_true
dL_da2=np.dot(dL_dz3,layer3.weights.T)
dL_dz2=dL_da2*(z2>0)
dL_da1=np.dot(dL_dz2,layer2.weights.T)
dL_dz1=dL_da1*(z1>0)

dw3=np.dot(a2.T,dL_dz3)
dw2=np.dot(a1.T,dL_dz2)
dw1=np.dot(train_images.T,dL_dz1)

layer3.weights-=(learning_rate*dw3)
layer2.weights-=(learning_rate*dw2)
layer1.weights-=(learning_rate*dw1)
layer3.biases-=(learning_rate*np.sum(dL_dz3,axis=0,keepdims=True))
layer2.biases-=(learning_rate*np.sum(dL_dz2,axis=0,keepdims=True))
layer1.biases-=(learning_rate*np.sum(dL_dz1,axis=0,keepdims=True))

# Finish the run and upload any remaining data.
run.finish()