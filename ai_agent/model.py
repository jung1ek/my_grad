from tensor.nn import Module,LinearLayer
from tensor.function import Relu

class QNet(Module):

    def __init__(self,input_size=11,output_size=3):
        super().__init__()
        self.layer1 = LinearLayer(in_features=input_size,out_features=32,act=Relu)
        self.output_layer = LinearLayer(in_features=32,out_features=output_size,act=None)

    def forward(self, x):
        output = self.layer1(x)
        output = self.output_layer(output)
        return output


if __name__=='__main__':
    model =QNet()
    x = [1,0,1,0,1,0,0,1,0,0,0]
    output = model(x)
    print(output)

class QTrainer:

    def __init__(self):
        pass

    def train_step(self):
        pass