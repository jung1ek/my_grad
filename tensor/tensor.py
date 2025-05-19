# From karpathy implementation we just implement Context (ctx), inplace for _prev/children
""" 1. The forward method computes the operation's output.
    2. The backward method computes the gradients of the inputs based on gradients of the output. (chain rule)
    Actual pytorch , all operation are performed by C++
"""
# from helpers import Context
try:
    # First try relative import (works when running as package)
    from .helpers import Context
except ImportError:
    # Fall back to direct import (works when running file directly)
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
        
        inputs = [inp if isinstance(inp, Tensor) or type(inp)==list else Tensor(inp) for inp in inputs]  # validate the *inputs (which is a and b), if input is list ignore
        output_data = cls.forward(ctx,*inputs) # operation; * sends arguments as tuple

        output = Tensor(output_data) # eg: output = f, f(a,b) = a*b
        output.grad_fn = cls # Link the Function to the output Tensor
        output.ctx = ctx  # Save the context for backward, like children, (a,b) for f
        output.is_leaf = False  # Result of an operation, not a leaf node
        return output
    
try:
    # First try relative import (works when running as package)
    from . import function as F
except ImportError:
    # Fall back to direct import (works when running file directly)
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

    def backward(self):
        """"Building the topo list containing the all the function in reversed order"""
        topo = [] # topological sorting: stores nodes(only function, no leaf) in reversed topological order
        visited = set() # a set to track visited nodes; ensuring we dont visit the same node multiple times.
        def build_topo(v): #recursive function to perform DFS and build topological order.
          if v not in visited: # Check if the current node has already been visited. (to make sure same function doesnot get repeated in topo list)
            visited.add(v) # mark the current node as visited

            if not v.is_leaf: # if the node is not a leaf (i.e> it ia an intermediate computation f= a*b (f) (a and b; leaf)
                for child in v.ctx.saved_tensors: # iterate over the child node stored in context
                  build_topo(child) # recursively build the topological order for the child nodes. DFS
                topo.append(v) # after the visiting and adding all the child nodes; then the current node is added the 'topo' list.
        build_topo(self)

        """Compute gradients by traversing the computation graph."""
        self.grad = 1.0 # set the gradient of the final output or final function to 1.0; which is df/df = 1.0
        for node in reversed(topo): # iterate over the nodes in reversed topological order.
            output_grad = node.grad # retrive the gradient of the current node from previous computation.
            node.grad_fn.backward(node.ctx,output_grad) # call the backward method of the function (chain rule)

    def __mul__(self,other): # f = a*b, self=a, other=b, f is return by Mul.apply method, which is output (f)
        """Overload the * operator."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return F.Mul.apply(self,other)
    
    def __add__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return F.Add.apply(self,other)
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return F.Sub.apply(self, other)
    
    def __pow__(self,other):
        other= other if isinstance(other, Tensor) else Tensor(other)
        return F.Pow.apply(self, other)
    
    def __truediv__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return F.Div.apply(self,other)
    
    def __neg__(self):
        return F.Neg.apply(self)

    def tanh(self):
        return F.Tanh.apply(self)
    
    def sigmoid(self):
        return F.Sigmoid.apply(self)
    
    def relu(self):
        return F.Relu.apply(self)
    
    def exp(self):
        return F.Exp.apply(self)
    
    def sqrt(self):
        return self.__pow__(1/2)
    
    def __ge__(self,other):
        if self.data>=other.data:
            return True
        else:
            return False
        
#     def __call__(self):
#         return self.data

    def __repr__(self):
        return f"Tensor({self.data}, grad_fn=<{self.grad_fn.__name__ if self.grad_fn else None}Backward>)"
    