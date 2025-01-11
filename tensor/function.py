from tensor import Function
class Multiply(Function):
    @staticmethod
    def forward(ctx,a,b): # implement mul operation and save the ctx
        """Save inputs for backward pass. and perform  multiplication. Got invoked in Funciton.apply method"""
        ctx.save_for_backward(a,b)
        return a.data * b.data
    
    @staticmethod
    def backward(ctx,outputs_grad):
        """Compute gradients for inputs. Got invoked in Tensor.backward method"""
        x,y = ctx.saved_tensors
        """ op = (f = a*b) ctx contains (a,b) and outputs_grad is the grad of the f,
            df/da = b and df/db = a, 
            chain rule, we multiply with f.grad ( which is dg/df) eg: g= f*.. ,
            graient from the last function to a and b are a.grad and b.grad
        """
        x.grad += y.data * outputs_grad # multivariative so we need to add the grad. x.grad+=
        y.grad += x.data * outputs_grad # multivariative so we need to add the grad. x.grad+=


class Add(Function):
    @staticmethod
    def forward(ctx,a,b):
        ctx.save_for_backward(a,b)
        return a.data+b.data
    @staticmethod
    def backward(ctx,output_grad):
        a,b =ctx.saved_tensors
        a.grad+=1*output_grad
        b.grad+=1*output_grad

def exp(x):
    e = 2.718281 # eular value
    return e**x

class Tanh(Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return (exp(2*x.data)-1)/(exp(2*x.data)+1)
    

    @staticmethod
    def backward(ctx,output_grad):
        x, = ctx.saved_tensors
        op = (exp(2*x.data)-1)/(exp(2*x.data)+1)
        x.grad+= (1-op**2)*output_grad # dtanh/dx= 1-tahn(x)**2
