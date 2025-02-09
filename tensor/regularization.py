from tensor import Function, Tensor
import numpy as np

class L2Reg(Function):

    def __call__(self, parameters, alpha):
        """
        Computes the Ridge Regression.
        Args:
            parameters: List of  trainable parameters (list of Tensor).
            alpha : regularization strength  (Tensor).
        Returns:
            The computed l2 Regularization Tenosr.
        """
        # Ensure inputs are of the correct type
        alpha = Tensor(alpha)
        assert type(alpha) == Tensor and type(parameters) == list
        
        # Apply the forward pass using the MCELoss function
        return L2Reg.apply(parameters, alpha)

    @staticmethod
    def forward(ctx,parameters,alpha):
        ctx.save_for_backward(*parameters,alpha) # tuple contains (w1,..,wn,alpha)
        sigma_square = sum(w.data**2 for w in parameters) # sigma(1..n( wi**2))
        return 1/2*alpha.data*sigma_square # 1/2*delta*Sigma(wi)

    @staticmethod
    def backward(ctx,output_grad):
        parameters = ctx.saved_tensors[:-1]# w1..wn
        alpha = ctx.saved_tensors[-1] # alpha
        for w in parameters:
            if w.requires_grad:
                w.grad+= alpha.data * w.data* output_grad # dL2/dw = alpha*w


class L1Reg(Function):

    def __call__(self, parameters, alpha):
        """
        Computes the L1 Regularization.
        Args:
            parameters: List of  trainable parameters (list of Tensor).
            alpha : regularization strength  (Tensor).
        Returns:
            The computed L1 Regularization Tenosr.
        """
        # Ensure inputs are of the correct type
        alpha = Tensor(alpha)
        assert type(alpha) == Tensor and type(parameters) == list
        
        # Apply the forward pass using the MCELoss function
        return L1Reg.apply(parameters, alpha)

    @staticmethod
    def forward(ctx,parameters,alpha):
        ctx.save_for_backward(*parameters,alpha)
        sigma_abs = sum(abs(w.data) for w in parameters)
        return alpha.data * sigma_abs

    @staticmethod
    def backward(ctx,output_grad):
        parameters = ctx.saved_tensors[:-1]
        alpha = ctx.saved_tensors[-1]
        for w in parameters:
            if w.requires_grad:
                w.grad+= output_grad*np.sign(w.data)*alpha.data
