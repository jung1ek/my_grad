from tensor import Function
from math import exp
class Mul(Function):
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
        return 1.0/(1.0+exp(-x.data))
    @staticmethod
    def backward(ctx,output_grad):
        x, = ctx.saved_tensors
        op = 1.0/(1.0+exp(-x.data))
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

class Softmax(Function):
    """
    Implements the softmax function for a single logit in a list of logits.
    Provides methods for both forward and backward passes to enable gradient computation.
    """
    
    @staticmethod
    def forward(ctx, current_logit, logits):
        """
        Forward pass for the softmax function.
        Args:
            ctx: Context to save variables for the backward pass.
            current_logit: The logit for which softmax is being computed (Tensor).
            logits: List of all logits (Tensor objects).
        Returns:
            Softmax output for the current_logit.
        """
        # Combine the current logit with the rest of the logits
        all_logits = [current_logit] + logits
        
        # Save all logits in the context for use in the backward pass
        ctx.save_for_backward(*all_logits)
        
        # Compute the sum of exponentials of all logits: sigma(exp(xi))
        sigma_exp = sum((exp(logit.data) for logit in logits))
        
        # Compute the softmax output for the current logit: exp(x)/sigma(exp(xi))
        softmax_output = exp(current_logit.data) / sigma_exp
        
        return softmax_output

    @staticmethod
    def backward(ctx, output_grad):
        """
        Backward pass for the softmax function.
        Computes gradients with respect to all logits.
        Args:
            ctx: Context containing saved tensors from the forward pass.
            output_grad: Gradient of the output from the next layer.
        """
        # Retrieve the current logit and all logits from the saved context
        current_logit = ctx.saved_tensors[0]
        
        # Compute the sum of exponentials of all logits: sigma(exp(xi))
        sigma_exp = sum((exp(logit.data) for logit in ctx.saved_tensors[1:]))
        
        # Compute the softmax value for the current logit: exp(x)/sigma(exp(xi))
        current_softmax = exp(current_logit.data) / sigma_exp

        # Iterate over all logits to compute their gradients
        for logit in ctx.saved_tensors[1:]:
            if logit.requires_grad:  # Only compute gradients for trainable logits
                
                if logit == current_logit:  # Diagonal case (i == j)
                    delta = 1  # Kronecker delta is 1 when i == j
                    logit_softmax = exp(logit.data) / sigma_exp  # Softmax of this logit
                    # Gradient: output_grad * softmax_i * (delta - softmax_j)
                    logit.grad += output_grad * (current_softmax * (delta - logit_softmax))
                else:  # Off-diagonal case (i != j)
                    delta = 0  # Kronecker delta is 0 when i != j
                    logit_softmax = exp(logit.data) / sigma_exp  # Softmax of this logit
                    # Gradient: output_grad * softmax_i * (delta - softmax_j)
                    logit.grad += output_grad * (current_softmax * (delta - logit_softmax))
