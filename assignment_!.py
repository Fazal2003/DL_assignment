import numpy as np

# making a neural network to classify the fashion mnsit data set 

class layer:
    def __init__(self,input_size,neurons):
        self.weights=np.random.randn(input_size,neurons)
        self.biases=np.random.randn(1,neurons)
        
    def forward(self,in_image):
        return np.dot(in_image,self.weights)+self.biases
    

class activation:
    def relu(self,z):
        self.z=z
        return np.maximum(0,self.z)
    
    def sgm(self,z):
        self.z=z
        return (1/(1+np.exp(-self.z)))
    
    def sftmx(self,z):
        self.z=z
        n=np.exp(z-(np.max(self.z)))
        d=np.sum(n)
        return n/d
    
class loss:
    def mse(self,yh,y):
        self.yh=yh
        self.y=y
        return (np.square(self.yh-self.y))
    
    def cross_entropy(self,yh,y):
        self.yh=yh
        self.y=y
        l=self.y*np.log(slef.y+1e-5)
        return -np.sum(l)
    

from keras.datasets import fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

size=len(train_images)
def one_hot_encode(labels,classes):
    one_hot=np.zeros((size,classes))
    one_hot[np.arange(labels.shape[0]),labels]=1
    return one_hot
y_true=one_hot_encode(train_labels)

n=3 #number of layers
layer1=layer(784,16)
layer2=layer(16,16)
layer3=layer(16,10)
activ=activation()
cost=loss()

images=(train_images-np.min(train_images))/(np.max(train_images)-np.min(train_images))

for i in range(size):
    image=images[i].reshape(1,784)

    z1=layer1.forward(image)
    a1=activ.relu(z1)
    z2=layer1.forward(a1)
    a2=activ.relu(z2)
    z3=layer1.forward(a2)
    a3=activ.sftmx(z3)
    a3=np.clip(a3,1e-7,1-1e-7)

    l=loss.cross_entropy(a3,y_true)



