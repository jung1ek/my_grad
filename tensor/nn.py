# Custom Tensor class, assumed to handle computations (e.g., gradients for backprop)
try:
    # First try relative import (works when running as package)
    from .tensor import Tensor
except ImportError:
    # Fall back to direct import (works when running file directly)
    from tensor import Tensor
import numpy as np
import math
from typing import List
try:
    # First try relative import (works when running as package)
    from .function import Relu,Sigmoid,Tanh,Relu,Softmax 
except ImportError:
    # Fall back to direct import (works when running file directly)
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
        k = 1/num_features
        bound= np.sqrt(k)
        random_data = np.random.uniform(-bound,bound,size=(num_features))  # Randomly initialize weights
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

try:
    # First try relative import (works when running as package)
    from . import function as F
except ImportError:
    # Fall back to direct import (works when running file directly)
    import function as F

def np_to_tensor():
    # elements of numpy array to Tensor object
    return np.vectorize(lambda x: Tensor(x))

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
        assert type(logits) == list or type(logits)==np.ndarray

        # apply layer norm over embed dimension; on embedding of each word.
        if type(logits)==np.ndarray and len(logits.shape)==2:
            outputs = [self.forward(x) for x in logits]
            return outputs
        else:
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
        # from pytorch random uniform distribution
        np_to_tensor = np.vectorize(lambda x: Tensor(x))  # Utility to convert numpy arrays to Tensors
        k = 1/(self.in_channels*self.kernel_size[0]*self.kernel_size[1])
        bound= np.sqrt(k)
        random_filters = np.random.uniform(-bound,bound,size=(self.out_channels, self.kernel_size[0], self.kernel_size[1]))
        self.filters = np_to_tensor(random_filters)  # Convert filters to Tensors

        # Initialize bias terms (one per output channel)
        self.bias = np_to_tensor(np.random.uniform(-bound,bound,size=(out_channels))).tolist()

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
        apply_tanh = np.vectorize(lambda x: Tanh.apply(x))
        return apply_tanh(x)
    
    def parameters(self):
        None

class Sigmoid(Sigmoid):

    def __call__(self,x):
        apply_sig = np.vectorize(lambda x: Sigmoid.apply(x))
        return apply_sig(x)
    
    def parameters(self):
        None

class RNN:
    
    def __init__(self,input_size,hidden_size,act=Tanh,output_layer=False):
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
        output = []
        for i in range(seq_len):
            h = self.act(
                np.dot(self.W,h)+self.b_hh+
                np.dot(self.U,x[i])+self.b_xh
            )
            output.append(h)
        # output; every hidden output , for each word in sequence.
        return output, h
    
    def parameters(self):
        # only python list.
        return self.W.ravel().tolist()+self.U.ravel().tolist()+self.b_hh.tolist()+self.b_xh.tolist()



class LSTM:

    def __init__(self,input_size,hidden_size,act=None):
        """input_size : The number of expected features in the input x
          hidden_size : The number of features in the hidden state h 
        """
        # forgot gate
        # input gate
        # output gate
        k = 1/hidden_size
        bound = np.sqrt(k)

        # where b, U and W respectively denote the biases, input weights and recurrent weights into the LSTM cell.
        random_wf = np.random.uniform(-bound,bound,(hidden_size,hidden_size))
        random_uf = np.random.uniform(-bound,bound,(hidden_size,input_size))
        random_bf = np.random.uniform(-bound,bound,size=(hidden_size))

        random_wg = np.random.uniform(-bound,bound,(hidden_size,hidden_size))
        random_ug = np.random.uniform(-bound,bound,(hidden_size,input_size))
        random_bg = np.random.uniform(-bound,bound,size=(hidden_size))

        random_wo = np.random.uniform(-bound,bound,(hidden_size,hidden_size))
        random_uo = np.random.uniform(-bound,bound,(hidden_size,input_size))
        random_bo = np.random.uniform(-bound,bound,size=(hidden_size))

        random_w = np.random.uniform(-bound,bound,(hidden_size,hidden_size))
        random_u = np.random.uniform(-bound,bound,(hidden_size,input_size))
        random_b = np.random.uniform(-bound,bound,size=(hidden_size))

        # weights for forgot gate. long term memory forgot old information.
        self.Uf = np_to_tensor()(random_uf)
        self.Wf = np_to_tensor()(random_wf)
        self.bf = np_to_tensor()(random_bf)

        # weights for input gate.
        self.Ug = np_to_tensor()(random_ug)
        self.Wg = np_to_tensor()(random_wg)
        self.bg = np_to_tensor()(random_bg)
        
        # weights output gate.
        self.Uo = np_to_tensor()(random_uo)
        self.Wo = np_to_tensor()(random_wo)
        self.bo = np_to_tensor()(random_bo)

        # weights for state gate, long term memory retain new info.
        self.U = np_to_tensor()(random_u)
        self.W = np_to_tensor()(random_w)
        self.b = np_to_tensor()(random_b)

        self.hidden_size = hidden_size
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self,x,s_o=None,h_o=None):
        seq_len, in_features = x.shape
        if s_o is None:
            s_o = np.zeros(self.hidden_size)

        if h_o is None:
            h_o = np.zeros(self.hidden_size)
        
        h = h_o  # Initialize short-term memory (hidden state)
        s = s_o  # Initialize long-term memory (cell state)
        output = []

        for i in range(seq_len):
            # Forget gate - decides how much of previous long-term memory to forget
            f = self.sigmoid(np.dot(self.Wf, h) + np.dot(self.Uf, x[i]) + self.bf)
            
            # Input gate - decides how much new information to write to long-term memory
            g = self.sigmoid(np.dot(self.Wg, h) + np.dot(self.Ug, x[i]) + self.bg)

            # Candidate memory content - new information to be potentially added to long-term memory
            candidate = self.sigmoid(np.dot(self.W, h) + np.dot(self.U, x[i]) + self.b)

            # Update long-term memory (cell state)
            # Forget old memory with 'f * s', and add new memory with 'g * candidate'
            s = f * s + g * candidate

            # Output gate - decides how much of the cell state should influence the short-term memory
            q = self.sigmoid(np.dot(self.Wo, h) + np.dot(self.Uo, x[i]) + self.bo)

            # Update short-term memory (hidden state)
            h = self.tanh(s) * q

            # Collect output for each time step
            output.append(h)

        # Return the sequence of hidden states, final hidden state, and final cell state
        return np.array(output), h, s


    def parameters(self):
        return (
            self.Uf.ravel().tolist() +
            self.Wf.ravel().tolist() +
            self.bf.ravel().tolist() +

            # Weights for input gate
            self.Ug.ravel().tolist() +
            self.Wg.ravel().tolist() +
            self.bg.ravel().tolist() +

            # Weights for output gate
            self.Uo.ravel().tolist() +
            self.Wo.ravel().tolist() +
            self.bo.ravel().tolist() +

            # Weights for state gate (long-term memory update)
            self.U.ravel().tolist() +
            self.W.ravel().tolist() +
            self.b.ravel().tolist()
        )


class GRU:

    def __init__(self,input_size,hidden_size):
        # update gate
        # reset gate
        k = 1/hidden_size
        bound = np.sqrt(k)
        random_wu = np.random.uniform(-bound,bound,(hidden_size,hidden_size))
        random_uu = np.random.uniform(-bound,bound,(hidden_size,input_size))
        random_bu = np.random.uniform(-bound,bound,size=(hidden_size))

        random_wr = np.random.uniform(-bound,bound,(hidden_size,hidden_size))
        random_ur = np.random.uniform(-bound,bound,(hidden_size,input_size))
        random_br = np.random.uniform(-bound,bound,size=(hidden_size))

        random_w = np.random.uniform(-bound,bound,(hidden_size,hidden_size))
        random_u = np.random.uniform(-bound,bound,(hidden_size,input_size))
        random_b = np.random.uniform(-bound,bound,size=(hidden_size))

        # weights for reset gate.
        self.Ur = np_to_tensor()(random_ur)
        self.Wr = np_to_tensor()(random_wr)
        self.br = np_to_tensor()(random_br)

        # weights for update gate.
        self.Uu = np_to_tensor()(random_uu)
        self.Wu = np_to_tensor()(random_wu)
        self.bu = np_to_tensor()(random_bu)
        
        # weights for state gate,
        self.U = np_to_tensor()(random_u)
        self.W = np_to_tensor()(random_w)
        self.b = np_to_tensor()(random_b)

        self.hidden_size = hidden_size
        self.sigmoid = Sigmoid()
        self.tanh = Sigmoid()


    def forward(self,x,h_o=None):
        seq_len,in_features = x.shape
        if h_o is None:
            h_o = np.zeros(self.hidden_size)
        h = h_o  # Initialize hidden state (short-term memory)
        output = []

        for i in range(seq_len):
            # Reset gate - controls how much of the previous hidden state to forget for candidate calculation
            r = self.sigmoid(np.dot(self.Wr, h) + np.dot(self.Ur, x[i]) + self.br)

            # Update gate - controls how much of the new state to keep vs how much of the previous hidden state to retain
            u = self.sigmoid(np.dot(self.Wu, h) + np.dot(self.Uu, x[i]) + self.bu)

            # Candidate state (proposed new hidden state)
            # Reset gate 'r' is applied to 'h' to forget some past information before combining with input
            new_state = self.tanh(np.dot(self.W, h) * r + np.dot(self.U, x[i]) + self.b)

            # Final hidden state: blend between the old hidden state and the candidate
            # If u ≈ 1 → keep mostly old hidden state; if u ≈ 0 → replace with new candidate
            h = (- u+1) * new_state + u * h

            # Store the current hidden state
            output.append(h)

        # Return the sequence of hidden states, and final hidden state
        return np.array(output), h

    def parameters(self):
        return (
            self.Ur.ravel().tolist() +
            self.Wr.ravel().tolist() +
            self.br.ravel().tolist() +

            # Weights for update gate
            self.Uu.ravel().tolist() +
            self.Wu.ravel().tolist() +
            self.bu.ravel().tolist() +

            # Weights for state gate
            self.U.ravel().tolist() +
            self.W.ravel().tolist() +
            self.b.ravel().tolist()
        )


class MultiHeadAttention:

    def __init__(self,embed_dim,num_heads=8,gain=1.0,mask=True):
        # linear projectile weights or can use linear layer
        # qw = np.random.randn(embed_dim,embed_dim)
        # kw = np.random.randn(embed_dim,embed_dim)
        # vw = np.random.randn(embed_dim,embed_dim)
        # ow = np.random.randn(embed_dim,embed_dim)

        k = 6/embed_dim*2
        bound= gain*np.sqrt(k)
        # input to hidden weights, size = (hidden_size, input_size)
        qw = np.random.uniform(-bound,bound,size=(embed_dim,embed_dim)) 
        kw = np.random.uniform(-bound,bound,size=(embed_dim,embed_dim)) 
        vw = np.random.uniform(-bound,bound,size=(embed_dim,embed_dim)) 
        ow = np.random.uniform(-bound,bound,size=(embed_dim,embed_dim)) 

        self.QW=np_to_tensor()(qw)
        self.KW=np_to_tensor()(kw)
        self.VW=np_to_tensor()(vw)
        self.OW=np_to_tensor()(ow)
        self.num_heads = num_heads
        self.d_model = embed_dim
        # d_k = d_model / h = 3 (each head processes some portion of embed value)
        self.dk = embed_dim//self.num_heads
        self.mask = mask

    def __call__(self,q,k,v):
        return self.forward(q,k,v)

    def forward(self,q,k,v):
        assert type(q)==np.ndarray and type(k)==np.ndarray and type(v)==np.ndarray,"input must be numpy array"
        seq_len,embed_dim = q.shape
        # linear transformation; linear projectile
        if q.dtype != 'O':
            q= np_to_tensor()(q)# when calling middle of the operation; create Tensor of data Tensor; Error;
        if k.dtype != 'O':
            k= np_to_tensor()(k)
        if v.dtype != 'O':
            v= np_to_tensor()(v)
        q_proj = np.dot(q,self.QW)
        k_proj = np.dot(k,self.KW)
        v_proj = np.dot(v,self.VW)

        # split into multiple heads
        # Reshape to (seq_len, h, d_k), then transpose to (h, seq_len, d_k)
        Q_heads = q_proj.reshape(seq_len,self.num_heads,self.dk).transpose(1,0,2)
        K_heads = k_proj.reshape(seq_len,self.num_heads,self.dk).transpose(1,0,2)
        V_heads = v_proj.reshape(seq_len,self.num_heads,self.dk).transpose(1,0,2)
        
        # for each head (boradcasting) avoid loops
        scores = np.matmul(Q_heads,K_heads.transpose(0,2,1),dtype=Tensor) # Q*Kt ;  matrix multiplication; shape(head, seq_len,seq_len)
        scores/= np.sqrt(self.dk) # scaling by sq-root(dk)

        # look-ahead mask, for scores.
        if self.mask:
            mask = np.triu(np.ones((self.num_heads,seq_len,seq_len)),k=1).astype(bool) # create the lookahead mask
            scores = np.where(mask,-np.inf,scores) # replace true with -inf value

            # solve the inf , float got no exp function error
            # other are Tensor so, try to call exp method on loop, and got error on inf value;
            inf_to_tensor = np.vectorize(lambda x: x if type(x)==Tensor else Tensor(x))
            scores = inf_to_tensor(scores)

        # in higher model_dimension (embedding_dim), and heads; the product is high and exp(high) becomes too high, overflow.
        #solution , initialize the weights based on pytorch implementation uniform(-bound,bound)

        # attention weights; softmax(scores); apply attention weights on value(v)
        # softmax along last axis shape(heads,seq,seq)
        #----> Method 1 <-------
        # Scores for Word 1 vs. Words 1, 2, 3; up to seq length
        attention_weights = np.exp(scores)/np.sum(np.exp(scores),axis=-1,keepdims=True)

        # #------> method 2 <------, using custom exp, same thing as using numpy exp
        # # Scores for Word 1 vs. Words 1, 2, 3; up to seq length, each row; is each softmax of one word
        # apply_exp = np.vectorize(lambda x: x.exp())
        # exp_scores = apply_exp(scores)  # Uses your Exp.apply()
        # sum_exp = exp_scores.sum(axis=-1, keepdims=True)
        # attention_weights = exp_scores / sum_exp
        # # print(attention_weights.sum(axis=-1,keepdims=True))

        # weighted sum of values
        # apply attention weights to each words, zi = summation(ai*z)
        atten_output = np.matmul(attention_weights,V_heads) # shape (heads,seq,dk)

        # concatenate heads and final projection; shape(seq,heads,dk)=>(seq,embed_dim)
        atten_output = atten_output.transpose(1,0,2).reshape(seq_len,self.d_model)
        return np.dot(atten_output,self.OW)
    
    def parameters(self):
        return self.QW.ravel().tolist() + self.KW.ravel().tolist()+ self.VW.ravel().tolist()+self.OW.ravel().tolist()


class PositionalEcnoding:

    def __init__(self,seq_len,embed_dim):
        # sin for even, cos for odd.
        # sine/cosine allow to learn relative pos; bounded value between [-1,1]
        self.PE = np.zeros((seq_len,embed_dim))
        self.position = np.arange(0,seq_len).reshape(seq_len,1) # pos, shape(seq_len,1)
        # div term actual formula (pos/10000^2i/embed_dim)
        # e −log(10000)⋅2i/d model; computationally efficient
        self.div_term = np.exp(np.arange(0,embed_dim,2,dtype=float)*(-math.log(10000.0)/embed_dim))
        # broadcasting for each position, even embed position.
        self.PE[:,0::2] = np.sin(self.position*self.div_term) # for even dim, shape(seq_len,(only even idx) embed_dim/2)
        self.PE[:,1::2] = np.cos(self.position*self.div_term) # for odd dim, shape(seq_len, (odd idx only) embed_dim+1/2)
        # if shape (3,4)
        # first pe = (3,2) # both even embed; 0,2 word in seq
        # second pe = (3,2) # both odd embed dim; 1,3 word in seq

    def __call__(self):
        return self.forward()

    def forward(self):
        # shape X (seq_len,embed_dim)
        pos_encoded = self.PE
        # if type(pos_encoded)==np.ndarray:
        #     return pos_encoded.tolist()
        return pos_encoded

    def parameters(self):
        None


class Embedding:

    def __init__(self,num_embeddings,embedding_size):
        # lookup table
        # num_embedding : dictionary size, embedding_size : size of each embedding vector
        np_weight = np.random.randn(num_embeddings,embedding_size)
        self.weight = np_to_tensor()(np_weight)
    
    def __call__(self,x):
        # apply to each element in input list.
        assert type(x)==list or type(x)==np.ndarray
        output_embed = [self.forward(xi) for xi in x]
        return np.array(output_embed)

    def forward(self,x):
        # x: indices
        # [i1,i2,i3]
        return self.weight[x]
    
    def parameters(self):
        return self.weight.ravel().tolist()
        
class TransformerEncoderLayer:
    def __init__(self, embed_dim, mlp_dim, vocab_size, num_heads):
        # Token embedding layer
        self.ie = Embedding(vocab_size, embed_dim)
        
        # Multi-head self-attention
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        
        # Layer normalization
        self.ln = LayerNorm(embed_dim)
        
        # Feedforward layers
        self.mlp_l1 = LinearLayer(embed_dim, mlp_dim)
        self.mlp_l2 = LinearLayer(mlp_dim, embed_dim, act=None)  # No activation in second layer
        
        # Store embedding dimension
        self.embed_dim = embed_dim

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        assert type(x) == np.ndarray, "input must be numpy array"
        seq_len, = x.shape

        if self.seq_len is None or self.seq_len != seq_len:
            self.seq_len = seq_len
        
        # Positional encoding initialized per sequence
        if self.pe is None:
            self.pe = PositionalEcnoding(self.seq_len, self.embed_dim)
        
        # Embed tokens and add positional encoding
        x = self.ie(x)
        x = x + self.pe()
        
        # Apply self-attention and residual connection
        x = self.ln(x + self.mha(x, x, x))
        
        # Feedforward block with residual + LayerNorm (applied per token)
        x = self.ln(x + np.array([self.mlp_l2(xi) for xi in [self.mlp_l1(xi) for xi in x]]))
        return x

    def parameters(self):
        # Return all model parameters
        return (
            self.ie.parameters()
            + self.mha.parameters()
            + self.ln.parameters()
            + self.mlp_l1.parameters()
            + self.mlp_l2.parameters()
        )

class TransformerDecoderLayer:
    def __init__(self, embed_dim, mlp_dim, vocab_size, num_heads):
        # Token embedding layer
        self.ie = Embedding(vocab_size, embed_dim)

        self.pe = None
        
        # Multi-head self-attention with masking (for autoregressive behavior)
        self.mask_mha = MultiHeadAttention(embed_dim, num_heads, mask=True)
        
        # Cross-attention over encoder output
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        
        # Layer normalization
        self.ln = LayerNorm(embed_dim)
        
        # Feedforward layers
        self.mlp_l1 = LinearLayer(embed_dim, mlp_dim)
        self.mlp_l2 = LinearLayer(mlp_dim, embed_dim, act=None)
        
        # Store embedding dimension
        self.embed_dim = embed_dim
        self.seq_len = None

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, Q, K, x):  # Q, K from encoder output; x is target tokens
        assert (
            type(Q) == np.ndarray and type(K) == np.ndarray and type(x) == np.ndarray
        ), "input must be numpy array"
        seq_len, = x.shape

        if self.seq_len is None or self.seq_len != seq_len:
            self.seq_len = seq_len
        
        # Positional encoding for target sequence
        if self.pe is None:
            self.pe = PositionalEcnoding(self.seq_len, self.embed_dim)
        
        # Embed target tokens and add positional encoding
        x = self.ie(x)
        x = x + self.pe()
        
        # Apply masked self-attention (prevents seeing future tokens)
        x = self.ln(x + self.mask_mha(x, x, x))
        
        # Cross-attention: decoder attends to encoder output
        x = np.array(x)  # Ensure x is a NumPy array
        x = self.ln(x + self.mha(Q, K, x))
        
        # Feedforward block with residual + LayerNorm
        x = self.ln(x + np.array([self.mlp_l2(xi) for xi in [self.mlp_l1(xi) for xi in x]]))
        return x

    def parameters(self):
        # Return all model parameters
        return (
            self.ie.parameters()
            + self.mha.parameters()
            + self.ln.parameters()
            + self.mlp_l1.parameters()
            + self.mlp_l2.parameters()
            + self.mask_mha.parameters()
        )

class Flatten:

    # 
    def __call__(self,x):
        return x.ravel().tolist()

    def parameters(self):
        None

