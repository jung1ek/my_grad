from tensor import Tensor  # Custom Tensor class, assumed to handle computations (e.g., gradients for backprop)
import numpy as np
from typing import List
from function import Relu,Sigmoid,Tanh,Relu,Softmax  # Activation functions, assumed to be implemented in `function` module


class Neuron:
    """Represents a single neuron in a neural network, with weighted inputs and an optional activation function."""
    
    def __init__(self, num_features, act=Relu):
        """
        Initialize a Neuron.
        Args:
            num_features: Number of input features (size of input vector).
            act: Activation function to be used (default: Relu).
        """
        random_data = np.random.rand(num_features)  # Randomly initialize weights
        self.w = [Tensor(xi) for xi in random_data]  # Convert weights to Tensor
        self.b = Tensor(1.0)  # Initialize bias as Tensor with value 1.0
        self.act = act  # Activation function

    def __call__(self, x):
        """
        Compute the forward pass for the neuron.
        Args:
            x: Input data (list of Tensor objects).
        Returns:
            Output after applying weights, bias, and activation function.
        """
        # Weighted sum (dot product) + bias
        x = [x] if np.isscalar(x) else x
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
        
    def forward(self,x):
        return self.__call__(x=x)


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
            Output after passing input to each neuron.
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
    
    def forward(self,x):
        return self.__call__(x=x)

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

import function as F
class LayerNorm:
    def __init__(self, features_dim, epsilon=1e-5, act=F.Relu):
        """
        Initializes the Layer Normalization module.

        Args:
            features_dim (int): The number of features (dimensions) in the input.
            epsilon (float, optional): A small constant added to the variance to avoid division by zero. Defaults to 1e-5.
            act (function, optional): Activation function to apply after normalization. Defaults to F.Relu.
        """
        self.features_dim = features_dim
        # Initialize gamma (scale) parameters as ones
        gamma_values = np.ones(features_dim)
        self.gamma = [Tensor(g) for g in gamma_values]
        # Initialize beta (shift) parameters as zeros
        beta_values = np.zeros(features_dim)
        self.beta = [Tensor(b) for b in beta_values]
        self.act = act  # Activation function
        self.epsilon = epsilon  # Small constant for numerical stability

    def __call__(self, logits):
        """
        Enables the instance to be called as a function, which calls the forward method.

        Args:
            logits (list): Input data to be normalized.

        Returns:
            list: Normalized and optionally activated output.
        """
        assert type(logits) == list
        return self.forward(logits)

    def forward(self, logits):
        """
        Forward pass for layer normalization.

        Args:
            logits (list): Input data to be normalized.

        Returns:
            list: Normalized and optionally activated output.
        """
        mu = np.mean(logits)  # Compute the mean of the input
        var = np.sum([(x - mu) ** 2 for x in logits]) / self.features_dim  # Compute the variance

        # Normalize the input using the mean and variance
        z = [(x - mu) / (var + self.epsilon) ** (1 / 2) for x in logits]

        # Scale and shift the normalized values using gamma and beta
        outputs = [gamma_i * z_i + beta_i for gamma_i, z_i, beta_i in zip(self.gamma, z, self.beta)]

        # Apply the activation function if provided
        if self.act:
            outputs = [self.act.apply(output) for output in outputs]

        return outputs

    def parameters(self):
        """
        Returns the trainable parameters of the layer (gamma and beta).

        Returns:
            list: List of gamma and beta parameters.
        """
        return self.gamma + self.beta



class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, act=Relu):
        """
        Initializes a 2D convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (number of filters).
            kernel_size (int or tuple): Size of the convolutional kernel (height, width).
            act (function, optional): Activation function to apply after convolution. Defaults to ReLU.
        """
        assert type(kernel_size) == tuple or type(kernel_size) == int
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # Activation function

        # Handle kernel_size input (convert int to tuple if necessary)
        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Initialize filters (kernels) with random values
        np_to_tensor = np.vectorize(lambda x: Tensor(x))  # Utility to convert numpy arrays to Tensors
        random_filters = np.random.rand(self.out_channels, self.kernel_size[0], self.kernel_size[1])
        self.filters = np_to_tensor(random_filters)  # Convert filters to Tensors

        # Initialize bias terms (one per output channel)
        self.bias = [Tensor(1) for _ in range(self.out_channels)]

    def __call__(self, x):
        """
        Enables the instance to be called as a function, which calls the forward method.

        Args:
            x (numpy.ndarray): Input image of shape (c, h, w).

        Returns:
            numpy.ndarray: Output feature map after convolution and activation.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Performs the forward pass of the convolutional layer.

        Args:
            x (numpy.ndarray): Input image of shape (c, h, w).

        Returns:
            numpy.ndarray: Output feature map of shape (out_channels, h', w').
        """
        assert len(x.shape) == 3, "Input image must be of shape (c, h, w)"

        # Calculate output shape
        output_shape = (
            self.out_channels,
            x.shape[1] - self.kernel_size[0] + 1,  # Height of output feature map
            x.shape[2] - self.kernel_size[1] + 1   # Width of output feature map
        )
        output = np.zeros(output_shape, dtype=object)  # Initialize output array

        # Perform convolution
        for i in range(output_shape[0]):  # Loop over output channels (filters)
            for j in range(output_shape[1]):  # Loop over height of output feature map
                for k in range(output_shape[2]):  # Loop over width of output feature map
                    # Extract the region of interest from the input
                    region = x[:, j:j + self.kernel_size[0], k:k + self.kernel_size[1]]
                    # Perform convolution operation (element-wise multiplication and summation)
                    conv_op = np.sum(self.filters[i] * region) + self.bias[i]
                    # Apply activation function if provided
                    if self.act:
                        conv_op = self.act.apply(conv_op)
                    # Store the result in the output array
                    output[i, j, k] = conv_op

        return output

    def parameters(self):
        """
        Returns the trainable parameters of the layer (filters and biases).

        Returns:
            list: List of filter and bias parameters.
        """
        parameters_list = self.filters.ravel().tolist()  # Flatten filters into a list
        return parameters_list + self.bias  # Combine filters and biases

class MaxPool2d:
    def __init__(self, kernel_size):
        """
        Initializes a 2D max pooling layer.

        Args:
            kernel_size (int or tuple): Size of the pooling window (height, width).
        """
        assert type(kernel_size) == tuple or type(kernel_size) == int
        # Handle kernel_size input (convert int to tuple if necessary)
        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

    def __call__(self, x):
        """
        Enables the instance to be called as a function, which calls the forward method.

        Args:
            x (numpy.ndarray): Input feature map of shape (c, h, w).

        Returns:
            numpy.ndarray: Output feature map after max pooling.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Performs the forward pass of the max pooling layer.

        Args:
            x (numpy.ndarray): Input feature map of shape (c, h, w).

        Returns:
            numpy.ndarray: Output feature map of shape (c, h', w').
        """
        assert len(x.shape) == 3, "Input features must be of shape (c, h, w)"

        # Calculate output shape
        output_shape = (
            x.shape[0],  # Number of channels remains the same
            x.shape[1] - self.kernel_size[0] + 1,  # Height of output feature map
            x.shape[2] - self.kernel_size[1] + 1   # Width of output feature map
        )
        output = np.zeros(output_shape, dtype=object)  # Initialize output array

        # Perform max pooling
        for i in range(output_shape[0]):  # Loop over channels
            for j in range(output_shape[1]):  # Loop over height of output feature map
                for k in range(output_shape[2]):  # Loop over width of output feature map
                    # Extract the region of interest from the input
                    region = x[i, j:j + self.kernel_size[0], k:k + self.kernel_size[1]]
                    # Perform max pooling operation
                    pool_op = np.max(region)
                    # Store the result in the output array
                    output[i, j, k] = pool_op

        return output

    def parameters(self):
        """
        Max pooling has no trainable parameters.

        Returns:
            None
        """
        None


class Relu(Relu):
    # Multi dimension implementation remains!!!
    def __call__(self,x):
        # using vectorzie 
        apply_relu = np.vectorize(lambda x: Relu.apply(x)) # apply function to each elements x in array
        return apply_relu(x) # return outputs
    
    def parameters(self):
        None

class Softmax(Softmax):

    def __call__(self,logits):
        probs = [] # probabilities of logits
        for logit in logits: # compute probability for each logit
            probs.append(Softmax.apply(logit,logits))
        return probs # return probabilites
    
    def parameters():
        None

class Tanh(Tanh):
    def __call__(self,x):
        apply_tanh = np.vectorize(lambda x: Tanh.apply(x) )
        return apply_tanh(x)
    
    def parameters(self):
        None

class RNN:
    
    def __init__(self,input_size,hidden_size,act=Tanh):
        # uniform distribution like pytorch RNN
        k = 1/hidden_size
        bound= np.sqrt(k)
        np_to_Tensor = np.vectorize(lambda x: Tensor(x))

        # input to hidden weights, size = (hidden_size, input_size)
        random_wu = np.random.uniform(-bound,bound,size=(hidden_size,input_size)) 
        self.U = np_to_Tensor(random_wu)

        # hidden to output weights, size = (output_size,hidden_size)
        self.V = None

        # hidden to hidden weights, size = (hidden_size, hidden_size)
        random_ww = np.random.uniform(-bound,bound,size=(hidden_size,hidden_size))
        self.W = np_to_Tensor(random_ww)

        # input to hidden bias, (hidden_size)
        random_bxh = np.random.uniform(-bound,bound,size=(hidden_size))
        self.b_xh = np_to_Tensor(random_bxh)

        # hidden to hidden bias, (hidden_size)
        random_bhh = np.random.uniform(-bound,bound,size=(hidden_size))
        self.b_hh = np_to_Tensor(random_bhh)

        self.act = act()
        self.hidden_size = hidden_size

    def __call__(self,x):
        return self.forward(x)

    def forward(self,x,h_0=None):
        # h_0 is initial hidden state
        # shape of x is (seq_len, features)
        seq_len,_ = x.shape
        if h_0 is None:
            h_0 = np.zeros(self.hidden_size)
        h = h_0

        # RNN implementation based on unfolded graph
        for i in range(seq_len):
            h = self.act(
                np.dot(self.W,h)+self.b_hh+
                np.dot(self.U,x[i])+self.b_xh
            )
        return h
    
    def parameters(self):
        # only python list.
        return self.W.ravel().tolist()+self.U.ravel().tolist()+self.b_hh.tolist()+self.b_xh.tolist()



class LSTM:

    def __init__(self):
        pass


class GRU:

    def __init__(self):
        pass



class Flatten:

    def __call__(self,x):
        return x.ravel().tolist()

    def parameters(self):
        None