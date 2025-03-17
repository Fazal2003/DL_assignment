import numpy as np

class activation:
    def relu(self,z):
        return np.maximum(0,z)
    
    def leaky_relu(self,z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    def back_relu(self, z, d_a):
        return d_a * (z > 0)
    
    def tanh(self,z):
        return np.tanh(z)

    def sgm(self,z):
        return (1/(1+np.exp(-z)))

    def back_sgm(self,z,d_a):
        return d_a*(self.sgm(z)*(1-self.sgm(z)))

    def sftmx(self,z):
        n=np.exp(z-np.max(z,axis=1,keepdims=True))
        d=np.sum(n,axis=1,keepdims=True)
        return n/d

    def back_sftmx(self,yh,y):
        return yh-y
    