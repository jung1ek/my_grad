class SGD:
    def __init__(self,parameters,lr=0.001):
        """
        parameters: list of weights Tensor
        """
        self.lr = lr
        self.parameters = parameters

    def step(self):
        pass
    def zero_grad(self):
        pass