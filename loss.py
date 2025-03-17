import numpy as np

class loss:
    def mse(self,yh, y):
        return (np.square(yh - y))

    def cross_entropy(self,yh, y):
        return -np.mean(np.sum(y * np.log(yh + 1e-8))  )
    
