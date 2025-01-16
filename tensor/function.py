from tensor import Function
from math import exp
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
        if x.requires_grad:
            x.grad += y.data * outputs_grad # multivariative so we need to add the grad. x.grad+=
        if y.requires_grad:
            y.grad += x.data * outputs_grad # multivariative so we need to add the grad. x.grad+=


class Add(Function):
    @staticmethod
    def forward(ctx,a,b):
        ctx.save_for_backward(a,b)
        return a.data+b.data
    @staticmethod
    def backward(ctx,output_grad):
        a,b =ctx.saved_tensors
        if a.requires_grad:
            a.grad+=1*output_grad
        if b.requires_grad:
            b.grad+=1*output_grad

class Tanh(Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return (exp(2*x.data)-1)/(exp(2*x.data)+1)
    

    @staticmethod
    def backward(ctx,output_grad):
        x, = ctx.saved_tensors
        op = (exp(2*x.data)-1)/(exp(2*x.data)+1)
        if x.requires_grad:
            x.grad+= (1-op**2)*output_grad # dtanh/dx= 1-tahn(x)**2

class Sigmoid(Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return 1.0/(1.0+math.exp(-x.data))
    @staticmethod
    def backward(ctx,output_grad):
        x, = ctx.saved_tensors
        op = 1.0/(1.0+math.exp(-x.data))
        if x.requires_grad:
            x.grad += (output_grad * op*(1.0-op))
    

class Relu(Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return max(0,x.data)
    @staticmethod
    def backward(ctx,output_grad):
        x, = ctx.saved_tensors
        if x.requires_grad:
            x.grad += (output_grad * (1.0 if x.data>0 else 0.0))
            

class Sub(Function):
    @staticmethod
    def forward(ctx, a,b):
        ctx.save_for_backward(a,b)
        return a.data + (-b.data)
    @staticmethod
    def backward(ctx, output_grad):
        x,y = ctx.saved_tensors
        if x.requires_grad:
            x.grad+= (output_grad*1.0)
            
        if y.requires_grad:
            y.grad+= (output_grad*(-1.0))
         
class Pow(Function):
    @staticmethod
    def forward(ctx, a, x):
        ctx.save_for_backward(a,x)
        return a.data**x
    @staticmethod
    def backward(ctx, output_grad):
        a,x = ctx.saved_tensors
        if a.requires_grad:
            a.grad += (output_grad * (x*a**(x-1.0)))
       
        ctx.saved_tensors = (a,)