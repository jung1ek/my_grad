from regularization import L2Reg
class SGD:
    def __init__(self,parameters,lr=0.001,momentum=0.9,weight_decay=0.01):
        """
        parameters: list of weights Tensor
        """
        self.lr = lr
        self.parameters = parameters

        # Momentum
        

    def step(self):
        for param in self.parameters:
            param.data -= self.lr*param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

class Adam:
    pass