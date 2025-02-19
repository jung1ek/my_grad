from tensor import Function,Tensor
import math
class MSELoss(Function):
    """
    Implements Mean Squared Error (MSE) Loss, commonly used for regression tasks.
    """

    def __call__(self, output, target):
        """
        Compute the MSE loss for given output and target.
        Args:
            output: Predicted output (Tensor).
            target: Ground truth values (can be Tensor or convertible to Tensor).
        Returns:
            MSE loss as a Tensor.
        """
        # Ensure the target is a Tensor
        target = target if isinstance(target, Tensor) else Tensor(target)
        # Apply the forward pass using the MSELoss function
        return MSELoss.apply(output, target)

    @staticmethod
    def forward(ctx, output, target):
        """
        Forward pass for the MSE loss.
        Args:
            ctx: Context to save variables for the backward pass.
            output: Predicted output (Tensor).
            target: Ground truth values (Tensor).
        Returns:
            Squared error between the output and target.
        """
        # Save output and target in the context for the backward pass
        ctx.save_for_backward(output, target)
        # Compute the squared error: (output - target)^2
        return (output.data - target.data) ** 2

    @staticmethod
    def backward(ctx, output_grad):
        """
        Backward pass for the MSE loss.
        Computes the gradient with respect to the output.
        Args:
            ctx: Context containing saved variables from the forward pass.
            output_grad: Gradient of the output from the next layer.
        """
        # Retrieve saved tensors (output and target) from the context
        output, target = ctx.saved_tensors
        
        # Compute gradient only if the output requires it
        if output.requires_grad:
            # Gradient: 2 * (output - target) * output_grad
            output.grad += output_grad * (2 * (output.data - target.data))


class BCELoss(Function):
    """
    Implements Binary Cross-Entropy (BCE) Loss, commonly used for binary classification tasks.
    """

    def __call__(self, output, target):
        """
        Compute the BCE loss for given output and target.
        Args:
            output: Predicted output or probability (Tensor).
            target: Ground truth values (can be Tensor or convertible to Tensor).
        Returns:
            BCE loss as a Tensor.
        """
        # Ensure the target is a Tensor
        target = target if isinstance(target, Tensor) else Tensor(target)
        # Apply the forward pass using the BCELoss function
        return BCELoss.apply(output, target)

    @staticmethod
    def forward(ctx, output, target):
        """
        Forward pass for the BCE loss.
        Args:
            ctx: Context to save variables for the backward pass.
            output: Predicted probability (Tensor).
            target: Ground truth binary values (Tensor).
        Returns:
            BCE loss value.
        """
        # Save output and target in the context for the backward pass
        ctx.save_for_backward(output, target)
        # Compute BCE loss: -[y * log(p) + (1-y) * log(1-p)]
        return -target.data * math.log(output.data) + (1 - target.data) * math.log(1 - output.data)

    @staticmethod
    def backward(ctx, output_grad):
        """
        Backward pass for the BCE loss.
        Computes the gradient with respect to the output.
        Args:
            ctx: Context containing saved variables from the forward pass.
            output_grad: Gradient of the output from the next layer.
        """
        # Retrieve saved tensors (output and target) from the context
        output, target = ctx.saved_tensors
        
        # Compute gradient only if the output requires it
        if output.requires_grad:
            # Gradient: (output - target) / (output * (1 - output)) * output_grad
            output.grad += output_grad * ((output.data - target.data) / (output.data * (1 - output.data)))


class MCELoss(Function):
    """
    Multi-Class Cross-Entropy Loss (MCE Loss).
    Commonly used for classification tasks with multiple classes.
    """

    def __call__(self, softmax_outputs, one_hot_label):
        """
        Computes the multi-class cross-entropy loss.
        Args:
            softmax_outputs: List of softmax outputs (list of Tensor).
            one_hot_label: One-hot encoded ground truth labels (Tensor).
        Returns:
            The computed MCE loss.
        """
        # Ensure inputs are of the correct type
        one_hot_label = Tensor(one_hot_label)
        assert type(one_hot_label) == Tensor and type(softmax_outputs) == list
        
        # Apply the forward pass using the MCELoss function
        return MCELoss.apply(softmax_outputs, one_hot_label)

    @staticmethod
    def forward(ctx, softmax_outputs, one_hot_label):
        """
        Forward pass for the MCE loss.
        Args:
            ctx: Context to save variables for the backward pass.
            softmax_outputs: List of softmax outputs (list of Tensor).
            one_hot_label: One-hot encoded ground truth labels (Tensor).
        Returns:
            The computed cross-entropy loss.
        """
        # Save all softmax outputs and the one-hot label in the context for the backward pass
        ctx.save_for_backward(*softmax_outputs, one_hot_label) # stores all values in single tuple(softmax,..,one_hot)
        
        # Compute the cross-entropy loss:
        # loss = -Σ(label * log(softmax_output))
        loss = -sum(
            (label * math.log(softmax.data) for label, softmax in zip(one_hot_label.data, softmax_outputs))
        )
        
        return loss

    @staticmethod
    def backward(ctx, output_grad):
        """
        Backward pass for the MCE loss.
        Computes the gradient with respect to each softmax output.
        Args:
            ctx: Context containing saved variables from the forward pass.
            output_grad: Gradient of the output from the next layer.
        """
        # Retrieve the one-hot encoded label from the context (last saved tensor)
        one_hot_label = ctx.saved_tensors[-1]
        
        # Compute the gradient for each softmax output
        for label, softmax in zip(one_hot_label.data, ctx.saved_tensors[:-1]):
            if softmax.requires_grad:  # Compute gradient only for trainable softmax outputs
                # Gradient: output_grad * (softmax_output - label)
                softmax.grad += output_grad * (softmax.data - label)
