""" 1. The forward method computes the operation's output.
    2. The backward method computes the gradients of the inputs based on gradients of the output. (chain rule)
    Actual pytorch , all operation are performed by C++
"""
from context import Context
class Function:
    @staticmethod
    def forward(ctx,*tensors):
        """Define the forward pass. Must be implemented by subclasses."""
        raise  NotImplementedError
        
    @staticmethod
    def backward(ctx,*outputs_grad):
        """Define the backward pass. Must be implemented by subclasses."""
        raise NotImplementedError
    
    """ This method get invoked while doing f=a*b
        f= output, (a,b) == (self,other)
    """ 
    @classmethod # apply multiply operation from forward method, set gradient_fn, set context, so-on
    def apply(cls,*inputs): # cls refers to Function class
        """Perform forward computation and link to backward."""
        ctx = Context() # iniatialize the context

        from tensor import Tensor # to solve circular import error. we are importing Tensor which, in first line import funcion module, that module
        # also import tensor module which is incomplete at that run-time.
        inputs = [inp if isinstance(inp, Tensor) else inp for inp in inputs]  # validate the *inputs (which is a and b)
        output_data = cls.forward(ctx,*inputs)

        output = Tensor(output_data)
        output.grad_fn = cls # Link the Function to the output Tensor
        output.ctx = ctx  # Save the context for backward
        output.is_leaf = False  # Result of an operation, not a leaf node
        return output

class Multiply(Function):
    @staticmethod
    def forward(ctx,a,b): # implement mul operation and save the ctx
        """Save inputs for backward pass. and perform  multiplication"""
        ctx.save_for_backward(a,b)
        return a.data * b.data
    
    @staticmethod
    def backward(ctx,outputs_grad):
        """Compute gradients for inputs."""
        x,y = ctx.saved_tensors
        """ op = (f = a*b) ctx contains (a,b) and outputs_grad is the grad of the f,
            df/da = b and df/db = a, 
            chain rule, we multiply with f.grad ( which is dg/df) eg: g= f*.. ,
            graient from the last function to a and b, a.grad and b.grad
        """
        x.grad += y.data * outputs_grad # multivariative so we need to add the grad. x.grad+=
        y.grad += x.data * outputs_grad # multivariative so we need to add the grad. x.grad+=
    

