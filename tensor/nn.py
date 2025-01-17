from tensor import Tensor  # Custom Tensor class, assumed to handle computations (e.g., gradients for backprop)
import numpy as np
from typing import List
from function import Relu, Sigmoid, Tanh  # Activation functions, assumed to be implemented in `function` module


class Neuron:
    """Represents a single neuron in a neural network, with weighted inputs and an optional activation function."""
    
    def __init__(self, num_features, act=Relu):
        """
        Initialize a Neuron.
        Args:
            num_features: Number of input features (size of input vector).
            act: Activation function to be used (default: Relu).
        """
        np.random.seed(69)  # Ensure reproducibility
        random_data = np.random.rand(num_features)  # Randomly initialize weights
        self.w = [Tensor(xi) for xi in random_data]  # Convert weights to Tensor
        self.b = Tensor(1.0)  # Initialize bias as Tensor with value 1.0
        self.act = act  # Activation function

    def __call__(self, x: List):
        """
        Compute the forward pass for the neuron.
        Args:
            x: Input data (list of Tensor objects).
        Returns:
            Output after applying weights, bias, and activation function.
        """
        # Weighted sum (dot product) + bias
        x = x if type(x) is list else [x] 
        output = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        
        # Apply activation function if provided
        if self.act:
            output = self.act.apply(output)
        
        return output
    
    def parameters(self):
        """
        Return all trainable parameters of the neuron (weights and bias).
        """
        return self.w + [self.b]
    
    def __repr__(self):
        """
        String representation of the neuron.
        """
        return f"{self.parameters()}"


class LinearLayer:
    """Represents a fully connected layer composed of multiple neurons."""
    
    def __init__(self, in_features, out_features, act=Relu): 
        """
        Initialize a Linear Layer.
        Args:
            in_features: Number of input features per neuron.
            out_features: Number of neurons in the layer.
            act: Activation function to be used (default: Relu).
        """
        self.linear_layer = [Neuron(num_features=in_features, act=act) for _ in range(out_features)]
        self.act = act  # Activation function for the layer
        self.__name__ = "Linear Layer"

    def __call__(self, x):
        """
        Compute the forward pass for the layer.
        Args:
            x: Input data (list of Tensor objects).
        Returns:
            Output after passing input through each neuron.
        """
        output = [neuron(x) for neuron in self.linear_layer]
        # If the layer has a single neuron, return a scalar instead of a list
        return output[0] if len(output) == 1 else output

    def parameters(self):
        """
        Return all trainable parameters of the layer (weights and biases for all neurons).
        """
        return [p for neuron in self.linear_layer for p in neuron.parameters()]

    def __repr__(self):
        """
        String representation of the layer.
        """
        return f"{self.__name__} Activation: {self.act}"


class MLP:
    """Represents a Multi-Layer Perceptron (MLP), composed of multiple fully connected layers."""
    
    def __init__(self, in_features, layers=()):
        """
        Initialize an MLP.
        Args:
            in_features: Number of input features to the network.
            layers: Tuple specifying the number of neurons in each layer.
        """
        self.layer_dims = (in_features,) + layers  # Include input features in layer dimensions
        self.layers = [
            LinearLayer(self.layer_dims[i], self.layer_dims[i + 1], act=None)
            if i == len(layers) - 1  # No activation on the last layer
            else LinearLayer(self.layer_dims[i], self.layer_dims[i + 1], act=Relu)  # Use ReLU for intermediate layers
            for i in range(len(layers))
        ]

    def __call__(self, x):
        """
        Compute the forward pass for the MLP.
        Args:
            x: Input data (list of Tensor objects).
        Returns:
            Output after passing input through all layers.
        """
        output = x
        for linear_layer in self.layers:
            output = linear_layer(output)
        return output

    def parameters(self):
        """
        Return all trainable parameters of the MLP (weights and biases for all layers).
        """
        return [p for linear in self.layers for p in linear.parameters()]

    def __repr__(self):
        """
        String representation of the MLP.
        """
        return f"{[linear for linear in self.layers]}"

class Module:
    """
    Base class for all neural network modules.
    Provides a consistent interface for creating and managing neural network components.
    """
    
    def __init__(self):
        """
        Initialize the Module.
        This base class does not initialize anything itself but serves as a parent class for other modules.
        """
        pass

    def __call__(self, x):
        """
        Enables the object to be called like a function.
        Args:
            x: Input data.
        Returns:
            Output from the `forward` method.
        """
        return self.forward(x)
    
    def forward(self, x):
        """
        Defines the forward pass for the module.
        This method must be implemented by any subclass.
        Args:
            x: Input data.
        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError
    
    def parameters(self):
        """
        Collects and returns all trainable parameters of the module.
        Assumes that trainable components (e.g., layers) are stored as attributes of the module.
        Returns:
            List of parameters from all trainable components of the module.
        """
        # Iterate over all attributes in the module and collect their parameters
        return [p for layer in self.__dict__.values() for p in layer.parameters()]
