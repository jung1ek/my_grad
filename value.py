class Value:
    def __init__(self,data, children=()):
        self._data = data
        self._prev = set(children)
        self.grad = 0.0 # gradient of final output with respect to self
        self._backward = lambda: None # chain rule function
    
    def __mul__(self, other): # fx = a: self * b: other, invoked by a (self)
        op = self._data * other._data
        fx = Value(op, (self,other))
        def backward(): # can only be invoked after the last function start invoking.
            self.grad += fx.grad * other._data
            other.grad += fx.grad * self._data
        fx._backward = backward
        return fx
    
    def __add__(self, other): # fx = a + b
        op = self._data + other._data
        fx = Value(op, (self,other))
        def backward(): # for fx, by self, set (grad of self and other with respect to Last function)
            self.grad+=fx.grad * 1.0
            other.grad+=fx.grad * 1.0
        fx._backward = backward
        return fx
    
    def __pow__(self, x: int)->Value:
        op = self._data**x
        fx = Value(op,(self,))
        def backward():
            self.grad += fx.grad * (x*self._data**(x-1))
        fx._backward = backward
        return fx
    
    def __sub__(self, other):
        op = self._data-other._data
        fx = Value(op,(self,other))
        def backward():
            self.grad += 1*fx.grad
            other.grad += (-1)*fx.grad
        fx._backward = backward
        return fx
    
    def relu(self):
        op = max(0,self._data)
        fx = Value(op,(self,))
        def backward():
            if self._data > 0:
                self.grad += 1.0*fx.grad
            else:
                self.grad += 0.0*fx.grad
        fx._backward = backward
        return fx
    
    def exp(self,x):
        e = 2.718281
        return e**x
    
    def tanh(self):
        x = self._data
        op = (self.exp(2*x)-1)/(self.exp(2*x)+1)
        fx = Value(op, (self,))
        def backward():
            self.grad+= (1-op**2)*fx.grad # dtahn/dx = 1-tanh^2(x)
        fx._backward = backward
        return fx
    
    def sigmoid(self):
        op = 1/(1+self.exp(-self._data))
        fx = Value(op,(self,))
        def backward():
            self.grad +=op*(1-op)*fx.grad # dsig/dx = sig(x).(1-sig(x))
        fx._backward = backward
        return fx

    def __repr__(self):
        return f"Value(data: {self._data})"
    
    def backward(self): # Chain rule
        self.grad = 1.0 # dfx/dfx
        
        stack = [self]
        while stack:
            current = stack.pop() # LIFO
            current._backward() # invoking the backward of last fx
            stack.extend(current._prev) # add children (inputs) of function to the stack