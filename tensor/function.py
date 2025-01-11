from tensor import Function
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
    

