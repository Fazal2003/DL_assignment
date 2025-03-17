import numpy as np

class optimzer:
  def __init__(self, weights, biases, learning_rate, grad_w,grad_b, beta=0.9,epsilon=1e-8):
    self.weights=weights
    self.biases=biases
    self.learning_rate=learning_rate
    self.grad_b=grad_b
    self.grad_w=grad_w
    self.beta=beta
    self.epsilon=epsilon
    self.vw=np.zeros_like(self.grad_w)
    self.vb=np.zeros_like(self.grad_b)
  
  def momentum(self):
    self.vw=np.zeros_like(self.grad_w)
    self.vb=np.zeros_like(self.grad_b)
    self.vw=self.beta*self.vw+(1-self.beta)*self.grad_w
    self.vb=self.beta*self.vb+(1-self.beta)*self.grad_b
    self.weights-=self.learning_rate*self.vw
    self.biases-=self.learning_rate*self.vb
    return self.weights,self.biases


  def nestrov(self):
    old_vw=self.vw
    old_vb=self.vb
    self.vw=self.beta*self.vw+(1-self.beta)*self.grad_w
    self.vb=self.beta*self.vb+(1-self.beta)*self.grad_b
    self.weights-=self.learning_rate*(self.beta*old_vw+(1-self.beta)*self.grad_w)
    self.biases-=self.learning_rate*(self.beta*old_vb+(1-self.beta)*self.grad_b)
        
    return self.weights,self.biases


  def adagrad(self):
    self.weights-=(self.learning_rate/(np.sqrt(self.grad_w**2)+self.epsilon))*self.grad_w
    self.biases-=(self.learning_rate/(np.sqrt(self.grad_b**2)+self.epsilon))*self.grad_b

    return self.weights,self.biases


  def rmsprop(self):
    self.sq_w = (self.beta*self.sq_w ) + ((1-self.beta)*(self.grad_w**2))
    self.sq_b = (self.beta*self.sq_v) + ((1-self.beta)*(self.grad_b**2))
    self.weigths-=(self.learning_rate/(np.sqrt(self.sq_w)+self.epsilon))*self.grad_w
    self.biases-=(self.learning_rate/(np.sqrt(self.sq_b)+self.epsilon))*self.grad_b
    
    return self.weights,self.biases