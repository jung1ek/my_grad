from tensor import Function,Tensor
import math
class MSELoss(Function):
    """For Regression"""
    def __call__(self,output,target):
        target = target if isinstance(target,Tensor) else Tensor(target)
        return MSELoss.apply(
            output,target
        )
    @staticmethod
    def forward(ctx,output,target):
        ctx.save_for_backward(output,target)
        return (output.data-target.data)**2
    @staticmethod
    def backward(ctx,output_grad):
        output,target = ctx.saved_tensors
        if output.requires_grad:
            output.grad += output_grad * (2*(output.data-target.data))

class BCELoss(Function):
    """For Binary Classification"""
    def __call__(self,output,target):
        target = target if isinstance(target,Tensor) else Tensor(target)
        return BCELoss.apply(output,target)

    @staticmethod
    def forward(ctx,output,target):
        ctx.save_for_backward(output,target)
        return -target.data*math.log(output.data)+(1-target.data)*math.log(1-output.data)
    @staticmethod
    def backward(ctx,output_grad):
        output,target = ctx.saved_tensors
        if output.requires_grad:
            output.grad+=output_grad *((output.data-target.data)/(output.data*(1-output.data)))