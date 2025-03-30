from regularization import L2Reg
class SGD:
    def __init__(self,parameters,lr=0.001,momentum=0.9,weight_decay=0.01):
        """
        parameters: list of weights Tensor
        """
        self.lr = lr
        self.parameters = parameters

        # Momentum
        self.velocity = [0 for _ in range(len(self.parameters))] #initial veloctiy is 0
        self.momentum = momentum

        # weight decay (not implemented)
        self.weight_decay =weight_decay

    def step(self):
        for param in self.parameters:
            param.data -= self.lr*param.grad
    
    def step_with_momentum(self):
        for i,param in enumerate(self.parameters):
            self.velocity[i] = self.velocity[i]*self.momentum-self.lr*param.grad
            param.data+=self.velocity[i]

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

class Adam:
    pass