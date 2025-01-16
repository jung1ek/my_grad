# From karpathy implementation we just implement Context (ctx), inplace for _prev/children
""" 1. The forward method computes the operation's output.
    2. The backward method computes the gradients of the inputs based on gradients of the output. (chain rule)
    Actual pytorch , all operation are performed by C++
"""
from helpers import Context
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

        inputs = [inp if isinstance(inp, Tensor) else inp for inp in inputs]  # validate the *inputs (which is a and b)
        output_data = cls.forward(ctx,*inputs) # operation


        output = Tensor(output_data) # eg: output = f, f(a,b) = a*b
        output.grad_fn = cls # Link the Function to the output Tensor
        output.ctx = ctx  # Save the context for backward, like children, (a,b) for f
        output.is_leaf = False  # Result of an operation, not a leaf node
        return output

import function as F
class Tensor:
    def __init__(self,data,requires_grad=True):
        self.data = data
        self.grad = None
        self.grad_fn = None # Stores the Function that created this Tensor
        self.is_leaf = True # True if this Tensor is not the result of an operation
        self.requires_grad = requires_grad
        self._version = 0 # prevent from the value change after.
        self.ctx = None # context refers to its children, f= a*b, f.ctx is (a and b), keep track of the computation graph
        self.set_grad()

    def set_grad(self):
        """Set the gradient to 0.0, if requires_grad"""
        if self.requires_grad:
            self.grad = 0.0
    
    @property
    def requires_grad(self):
        """Getter for requires_grad"""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self,value):
        """Setter for requires_grad that invokes set_grad"""
        self._requires_grad = value
        if value and self.grad is None:
            self.set_grad()
        elif not value and self.grad is not None:
            self.grad = None
    
    def __mul__(self,other): # f = a*b, self=a, other=b, f is return by Mul.apply method, which is output (f)
        """Overload the * operator."""
        return F.Mul.apply(self,other)
    
    def __add__(self,other):
        return F.Add.apply(self,other)

    def tanh(self):
        return F.Tanh.apply(self)

    def __repr__(self):
        return f"Tensor({self.data}, grad_fn=<{self.grad_fn.__name__}Backward>)"
    
    # we invoke this method from the last function so, eg: last funciton is f= a*b, in this case self=f
    def backward(self):
        """Compute gradients by traversing the computation graph."""
        self.grad = 1.0 # we df/df = 1.0, last node

        stack = [self] #LIFO
        while stack:
            current = stack.pop() # extract the last element or Tensor from stack, removes it too.
            output_grad = current.grad # set the output_grad from the current Tensor grad.

            """Add the children to the stack and invoke backward,f=current: f(a,b) = a and b are children. also, only if a or b are function too, not leaf"""
            # grad_fn is only in function (f), not in leaf (a, b)
            if current.grad_fn: # leaf doesnot contains grad_fn, f = a * b , a and b are leaf
                current.grad_fn.backward(current.ctx,output_grad)
            # ignore the leaf, leaf doesnot contains ctx, its None so,
            if current.is_leaf == False:
                stack.extend(current.ctx.saved_tensors)
