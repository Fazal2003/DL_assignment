import numpy as np

# making a neural network to classify the fashion mnsit data set

class layer:
    def __init__(self,input_size,neurons):
        self.weights=np.random.randn(input_size,neurons)
        self.biases=np.random.randn(1,neurons)
        self.layer_input=None


    def forward(self,in_image):
        self.layer_input=in_image
        return np.dot(in_image,self.weights)+self.biases




class activation:
    def relu(self,z):
        return np.maximum(0,z)

    def back_relu(self, z, d_a):
        return d_a.T * (z > 0)

    def sgm(self,z):
        return (1/(1+np.exp(-z)))

    def back_sgm(self,z,d_a):
        return d_a*(np.exp(z)*(1-np.exp(z)))


    def sftmx(self,z):
        n=np.exp(z-(np.max(z)))
        d=np.sum(n)
        return n/d

    def back_sftmx(self,yh,y):
        return (yh-y)


class loss:
    def mse(self,yh,y):
        return (np.square(self.yh-self.y))

    def cross_entropy(self,yh,y):
        l=y*np.log(yh+1e-5)
        return -np.sum(l)   



def momentum(weights,biases,dw,db,vw,vb,learning_rate,b=0.9):
    # vw=b*vw+(1-b)*dw
    # vb=b*vb+(1-b)*db
    # weights-=learning_rate*vw
    # biases-=learning_rate*vb
    # return weights,biases,vw,vb
    ut=0
    for i in range(steps):
      ut+=(b**(steps-i)+dw)
      weights-=learning_rate*ut
      return weights





def nestrov(weights,biases,db,dw,vw,vB,learning_rate,b=0.9):
    # uw=weights-b*vW
    # ub=biases-b*vB

    # vW = b*vW+(1-b)*dw
    # vB = b*vB+(1-b)*db
    # weights-=learning_rate*vW
    # biases-=learning_rate*vB
    # return weights,biases,vW,vB

    ut=0
    for i in range(steps):
      ut+=(b**(steps-i) +(weights-b*ut[i-1]))
      weights-=learning_rate*ut
      return weights

from keras.datasets import fashion_mnist
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
images=(train_images-np.min(train_images))/(np.max(train_images)-np.min(train_images))

#os 


for i in range(size):
        image=images[i].reshape(1,784)

        z1=layer1.forward(image)
        a1=activ.relu(z1)
        z2=layer2.forward(a1)
        a2=activ.relu(z2)
        z3=layer3.forward(a2)
        a3=activ.sftmx(z3)
        a3=np.clip(a3,1e-7,1-1e-7)

        l=cost.cross_entropy(a3,y_true[i])

        dL_dz3=a3-y_true[i]


        dL_da2=np.dot(dL_dz3,layer3.weights.T)
        dL_dz2=dL_da2*(z2>0)

        dL_da1=np.dot(dL_dz2,layer2.weights.T)
        dL_dz1=dL_da1*(z1>0)


        dw3=np.dot(a2.T,dL_dz3)
        dw2=np.dot(a2.T,dL_dz2)
        dw1=np.dot(image.T,dL_dz1)

        layer3.weights-=(learning_rate*dw3)
        layer2.weights-=(learning_rate*dw2)
        layer1.weights-=(learning_rate*dw1)
        layer3.biases-=(learning_rate*np.sum(dL_dz3,axis=0,keepdims=True))
        layer2.biases-=(learning_rate*np.sum(dL_dz2,axis=0,keepdims=True))
        layer1.biases-=(learning_rate*np.sum(dL_dz1,axis=0,keepdims=True))

        #momentum(layer1.weigths,dw1,db1)
        print(l)

  