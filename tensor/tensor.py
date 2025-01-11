# From karpathy implementation we just implement Context (ctx), inplace for _prev/children
from function import Multiply

class Tensor:
    def __init__(self,data,requires_grad=True):
        self.data = data
        self.grad = 0.0
        self.grad_fn = None # Stores the Function that created this Tensor
        self.is_leaf = True # True if this Tensor is not the result of an operation
        self.requires_grad = requires_grad
        self._version = 0 # prevent from the value change after.
        self.ctx = None # context refers to its children, f= a*b, f.ctx is (a and b), keep track of the computation graph
        
    def __mul__(self,other): # f = a*b, self=a, other=b, f is return by Mul.apply method, which is output (f)
        """Overload the * operator."""
        return Multiply.apply(self,other)

    def __repr__(self):
        return f"Tensor({self.data})"
    
    # we invoke this method from the last function so, eg: last funciton is f= a*b, in this case self=f
    def backward(self):
        """Compute gradients by traversing the computation graph."""
        self.grad = 1.0 # we df/df = 1.0

        stack = [self]
        while stack:
            current = stack.pop() # extract the last element or Tensor from stack, removes it too.
            output_grad = current.grad # set the output_grad from the current Tensor grad.
            
            # grad_fn is only in function (f), not in leaf (a, b)
            if current.grad_fn: # leaf doesnot contains grad_fn, f = a * b , a and b are leaf
                current.grad_fn.backward(current.ctx,output_grad)
            # ignore the leaf, leaf doesnot contains ctx, its None so,
            if current.is_leaf == False:
                stack.extend(current.ctx.saved_tensors)
