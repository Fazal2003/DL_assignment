import numpy as np
np.random.seed(42)

class layer:
    def __init__(self, input_size, neurons):
        self.weights = np.random.randn(input_size, neurons)
        self.biases = np.zeros((1, neurons))
        self.layer_input = None

    def forward(self, in_image):
        self.layer_input = in_image
        output=np.dot(in_image, self.weights) + self.biases
        output-np.nan_to_num(output)
        return output