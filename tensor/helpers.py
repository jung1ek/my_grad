""" 1. Saves tensors needed for gradient computation in ctx.save_for_backward.
    2. Retrieves saved tensors via ctx.saved_tensors.
   Computes the gradient with respect to the inputs.
"""
class Context:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self,*tensors): # *inputs, takes number of inputs in tuple , param = (a,b): inputs = (a,b), param=(a): inputs=(a,)
        """Save tensors for backward"""
        self.saved_tensors = tensors

    @property
    def saved_tensors(self):
        """Retrive saved tensors"""
        return self._saved_tensors
        
    @saved_tensors.setter
    def saved_tensors(self,tensors):
        self._saved_tensors = tensors