# -------------------------------------------------------------
# EDLA.py
#
# A PyTorch implementation of the Error‑Diffusion Learning Algorithm (EDLA).
#
# Author: Kazuhisa Fujita
# Since : 2024/09/03
# Update: 2025/04/20
# --------------------------------------------------------------
"""
Overview
--------
This module implements the Error-Diffusion Learning Algorithm (EDLA) using PyTorch. 
EDLA is a neural network learning algorithm that utilizes error diffusion for weight updates, originally proposed by Kaneko.

EDLA at a glance
----------------
* **Motivation** EDLA is a biologically-motivated alternative to backpropagation. It learns with a *single global error signal* that diffuses thoughout the network.
* **Network layout** Each *EDLayer* doubles the number of logical units so that every logical neuron is represented by a pair of positive and negative neurons in a positive sublayer and a negative sublayer, respectively. The weights between same types of neurons are excitatory, while the weights between different types of neurons are inhibitory.
* **Weight constraints** Connections are pre-initialised so that excitatory synapses are positive while inhibitory synapses are negative. Training preserves these sign constraints.
* **Learning rule** Weights update in proportion to the *sign* of the current weight, the presynaptic activation, the derivative of the postsynaptic activation, and the *global error* *d*.

File organisation
-----------------
1.  Utility functions
2.  Derivative helpers for activations and loss functions
3.  **EDLayer**    (implements one positive/negative dense block)
4.  **EDLA**       (EDLA model using the Error-Diffusion Learning Algorithm)
5.  **EDLA_Multi** (wrapper that hosts *K* independent EDLA heads so the model can emit multiple scalar outputs)

Notes
-----
- The EDLA and EDLA_Multi classes are designed for supervised learning tasks.
- Weight updates are performed manually using the error-diffusion principle, not standard PyTorch optimizers.
"""

# ============================
# Standard dependencies
# ============================
import torch
import torch.nn as nn
import torch.nn.init as init

# -------------------------------------------------------------
# Generic helper functions
# -------------------------------------------------------------

# Function to apply the selected activation function
def act_func(x, activation_fn="Sigmoid", negative_slope=0.01):
    if activation_fn == "Sigmoid":
        return torch.sigmoid(x)
    elif activation_fn == "ReLU":
        return torch.relu(x)
    elif activation_fn == "ReLU6":
        return torch.relu6(x)
    elif activation_fn == "LeakyReLU":
        return torch.leaky_relu(x, negative_slope=negative_slope)
    elif activation_fn == "Tanh":
        return torch.tanh(x)
    elif activation_fn == "None":
        return x
    else:
        print("Error")


# Function to compute the derivative of the activation function
def d_activation(x, activation_fn="Sigmoid", negative_slope=0.01, beta=1.0):

    if activation_fn == "Sigmoid":
        f = torch.sigmoid(x)
        return (1 - f)*f
    elif activation_fn == "ReLU":
        return (x > 0).float()
    elif activation_fn == "ReLU6":
        # Derivative is 1 for 0 < x < 6, else 0
        return ((x > 0) & (x < 6)).to(x.dtype)
    elif activation_fn == "LeakyReLU":
        # For LeakyReLU: derivative is 1 where x > 0, else negative_slope
        return torch.where(x > 0,
                        torch.ones_like(x),
                        torch.full_like(x, negative_slope))
    if activation_fn == "Tanh":
        return 1 - torch.tanh(x) ** 2
    elif activation_fn == "None":
        return torch.ones_like(x)
    else:
        print("Error")


# Function to compute the derivative of the loss function
def d_loss(y, t, loss_fc="MSE"):
    if loss_fc == "MSE":
        return y - t
    else:
        raise ValueError(f"Invalid loss_fc value: {loss_fc}. Supported value is 'MSE'.")


# ----------------------------------------------------------------
# Definition of the ED Layer
# ----------------------------------------------------------------
class EDLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(EDLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Define two fully connected layers representing the positive and negative sublayers.
        self.fc_p = nn.Linear(self.input_size, output_size*2) # Positive
        self.fc_n = nn.Linear(self.input_size, output_size*2) # Negative        

        # Initialize weights
        self.initialize_weights()

    # Initialize weights within specific ranges
    def initialize_weights(self):

        # Initialize weights and biases for the positive layer
        init.uniform_(self.fc_p.weight, a=-1, b=0)
        init.uniform_(self.fc_p.bias,   a=-1, b=0)

        # Set the first half of the weights and biases to positive values (w_pp >= 0)
        self.fc_p.weight.data[:self.output_size,:] *= -1
        self.fc_p.bias.data[:self.output_size]     *= -1

        # Initialize weights and biases for the negative layer
        init.uniform_(self.fc_n.weight, a=0, b=1)
        init.uniform_(self.fc_n.bias,   a=0, b=1)

        # Set the first half of the weights and biases to negative values (w_pn <= 0)
        self.fc_n.weight.data[:self.output_size,:] *= -1
        self.fc_n.bias.data[:self.output_size]     *= -1

    # Forward pass
    def forward(self, x):# x = [x_p, x_n]
        # Split input into excitatory and inhibitory parts, pass through respective layers, and sum results
        _, M = x.shape
        return self.fc_p(x[:,:M//2]) + self.fc_n(x[:,M//2:])


# --------------------------------------------------
# helper: holds 4 update buffers but勾配は追跡しない
# --------------------------------------------------
class _EDUpdateBuf(nn.Module):
    def __init__(self, layer: EDLayer):
        super().__init__()
        self.register_buffer("p_w", torch.zeros_like(layer.fc_p.weight))
        self.register_buffer("p_b", torch.zeros_like(layer.fc_p.bias))
        self.register_buffer("n_w", torch.zeros_like(layer.fc_n.weight))
        self.register_buffer("n_b", torch.zeros_like(layer.fc_n.bias))

    def zero_(self):
        for buf in self.buffers():
            buf.zero_()

# --------------------------------------------------------------
# EDLA
# EDLA model using the Error-Diffusion Learning Algorithm
# ---------------------------------------------------------------
class EDLA(nn.Module):
    """
    EDLA model using the Error-Diffusion Learning Algorithm.
    input_size:         Number of input features
    hidden_size:        Number of hidden neurons
    hidden_layers:      Number of hidden layers
    output_size:        Number of output neurons (1 recommended)
    activation_fn:      Activation function for hidden layers
    last_activation_fn: Activation function for output layer
    loss_fc:            Loss function ('MSE')
    learning_rate:      Learning rate for weight updates
    reduction:          Reduction method for weight updates ('mean' or 'sum')
    """

    def __init__(self,input_size, hidden_size, output_size, hidden_layers = 1, activation_fn = "Sigmoid", last_activation_fn = "Sigmoid", loss_fc="MSE", learning_rate=0.1, reduction='mean'):
        super(EDLA, self).__init__()

        # -- Store hyperparameters --
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_fn = activation_fn
        self.last_activation_fn = last_activation_fn
        self.loss_fc = loss_fc
        self.learning_rate = learning_rate        
        self.reduction = reduction

        # -- construct stack of layers --
        # input -> AF -> o -> AF -> ... -> o -> AF
        self.layers = nn.ModuleList()
        if hidden_layers != 0:
            self.layers.append(EDLayer(input_size, hidden_size))
            for i in range(hidden_layers-1):
                self.layers.append(EDLayer(hidden_size, hidden_size))
            self.layers.append(EDLayer(hidden_size, output_size))
        else:
            self.layers.append(EDLayer(input_size, output_size))

        self.num_layers = len(self.layers)

        # -- stack activation functions --
        # The first activation function is applied to the input but is "None"
        # The last layer has a different activation function
        self.activation_fns = []
        self.activation_fns.append("None")
        for i in range(self.num_layers-1):
            self.activation_fns.append(activation_fn)
        self.activation_fns.append(last_activation_fn)

        # Initialize buffer for weight updates
        # The buffer is used to store the weight updates for each layer
        self._dw_buf = nn.ModuleList([_EDUpdateBuf(lyr) for lyr in self.layers])
    
    # -- Forward pass --
    def forward(self, x):
        # Duplicate input for positive and negative sublayers
        # x = [x_p, x_n]
        # x_p = x_n = x        
        x = torch.cat([x, x], dim=1) 

        # Storing activation for each sublayer in input layer, hidden layers, and output layer
        # outputs = [input, activation of hidden layer 1, ..., activation of hidden layer N]
        outputs = [x]
        # Apply activation function to the input
        x = act_func(x, activation_fn=self.activation_fns[0])

        # Pass through each layer
        for l, layer in enumerate(self.layers):
            x = layer(x)
            outputs.append(x)
            x = act_func(x, activation_fn=self.activation_fns[l+1])

        # Split output into excitatory and inhibitory parts, and return the excitatory part
        # x = [x^L+, x^L-] -> x^L+, outputs = [input, activation of hidden layer 1, ..., activation of hidden layer N]
        
        _, M = x.shape
        return x[:,:M//2], outputs 


    # -- Computer differential weight --
    def dw(self, d, oi, oj, w):
        # d:  global error signal
        # oi: output of the current layer
        # oj: derivative of the next layer
        # w: weight of the current layer    
    
        dw = torch.matmul(oj, oi)

        # multiply of the sign of the weight
        # This multiplication defines the direction of the weight update
        dw *= torch.sign(w)
        dw *= d.unsqueeze(-1)
        return dw

    # -- Update model weights --
    def update(self, model, input_data, target):
        # Run forward pass
        o, outputs = model(input_data)

        # Compute the global error signal
        # The global error signal is negative derivative of the loss function
        diff = - d_loss(y=o, t=target, loss_fc=self.loss_fc)

        # Update weights
        self.update_core(model, outputs, diff)

    # -- Update model weights --
    def update_core(self, model, outputs, diff):
        # Initialize weight updates
        for buf in self._dw_buf:
            buf.zero_()

        # Loop through each layer and compute the weight updates
        for l in range(self.num_layers):
            layer = model.layers[l]
            buf = self._dw_buf[l]

            B, M = outputs[l].shape   # batch size, number of neurons in layer l
                                      # M = 2 * width of layer l (positive and negative sublayers)
            _, K = outputs[l+1].shape # batch size, number of neurons in layer l+1
                                      # K = 2 * width of layer l+1 (positive and negative sublayers)


            # ----- Avoid to change the sign of the weights -----
            # This process is not used in the current implementation
            # As long as using the Sigmoid and ReLU activation functions and the MSE loss function, signs of the weights are not changed.
            # layer.fc_p.weight.data[:M//2].clamp_(min=0)
            # layer.fc_p.weight.data[M//2:].clamp_(max=0)
            # layer.fc_n.weight.data[:M//2].clamp_(max=0)
            # layer.fc_n.weight.data[M//2:].clamp_(min=0)


            # ----- positive phase (d > 0) -----
            # d > 0 means that the output is smaller than the target when using the MSE loss function
            # Thus, we need to increase the positive neurons and decrease the negative neurons
            # Update the weight  from the positive sublayer
            # The positive sublayer is the first half of the weight matrix
        
            # Create a mask for samples where the global error signal is positive (i.e., output < target)
            # Set diff to zero for samples where the global error signal is negative
            # This is done to avoid updating the weights for these samples
            d_pos = diff.clamp_min(0)

            # Update the weight from the positive sublayer (not including the bias)
            # \eta * output of layer l * derivative of layer l+1
            buf.p_w.add_(self.learning_rate * torch.sum(
                self.dw(d_pos, act_func(outputs[l][:, :M//2].view(B, 1, M//2), activation_fn=self.activation_fns[l]),
                        d_activation(outputs[l+1].view(B, K, 1), activation_fn=self.activation_fns[l+1]), layer.fc_p.weight.data),
                dim=0))

            # Update the bias from the positive sublayer
            # \eta * output of layer l (an output of a bias neuron is 1) * derivative of layer l+1* derivative of layer l+1
            buf.p_b.add_(self.learning_rate * torch.sum(
                d_activation(outputs[l+1], activation_fn=self.activation_fns[l+1]) * torch.sign(layer.fc_p.bias.data)*d_pos,
                dim=0))

            # ----- negative phase (d < 0) -----
            # d < 0 means that the output is larger than the target when using the MSE loss function
            # Thus, we need to decrease the positive neurons and increase the negative neurons
            # Update the weight from the negative sublayer
            # The negative sublayer is the second half of the weight matrix

            # Set diff to zero for samples where the global error signal is posive
            # This is done to avoid updating the weights for these samples            
            d_neg = (-diff).clamp_min(0)

            # d < 0, g'(a^{p, (l)}_j) > 0, z^{n, (l-1)}_i > 0, sign(w^{nn, (l)}_{ji})> 0
            # dw^{nn, (l)}_{ji} = \eta (-d) * g'(a^{p, (l)}_j) z^{n, (l-1)}_i sign(w^{nn, (l)}_{ji}) > 0
            # Therefore, w^{nn, (l)}_{ji} is updated in positive direction.
            # sign(w^{pn, (l)}_{ji}) < 0
            # dw^{nn, (l)}_{ji} = \eta (-d) * g'(a^{p, (l)}_j) z^{n, (l-1)}_i sign(w^{pn, (l)}_{ji}) < 0
            # Therefore, w^{pn, (l)}_{ji} is updated in negative direction.
            buf.n_w.add_(self.learning_rate * torch.sum(
                self.dw(d_neg, act_func(outputs[l][:, M//2:].view(B, 1, M//2), activation_fn=self.activation_fns[l]),
                        d_activation(outputs[l+1].view(B, K, 1), activation_fn=self.activation_fns[l+1]), layer.fc_n.weight.data),
                dim=0))

            buf.n_b.add_(self.learning_rate * torch.sum(
                d_activation(outputs[l+1], activation_fn=self.activation_fns[l+1]) * torch.sign(layer.fc_n.bias.data)*d_neg,
                dim=0))

        # Update weights
        batch_size = diff.shape[0]
        for l in range(self.num_layers):
            layer = model.layers[l]
            buf = self._dw_buf[l]

            if self.reduction == 'mean':
                scale = 1.0 / batch_size
            elif self.reduction == 'sum':
                scale = 1.0
            else:
                raise ValueError(f"Invalid reduction method: {self.reduction}. Supported values are 'mean' and 'sum'.")

            # --- excitatory weights & bias
            layer.fc_p.weight.add_(buf.p_w * scale)
            layer.fc_p.bias  .add_(buf.p_b * scale)
            
            # --- inhibitory weights & bias
            layer.fc_n.weight.add_(buf.n_w * scale)
            layer.fc_n.bias  .add_(buf.n_b * scale)
              



# --------------------------------------------------------------
# EDLA_Multi - simple multi‑head wrapper
# Multi-output EDLA model combining multiple EDLA networks
# --------------------------------------------------------------
class EDLA_Multi(nn.Module):
    """
    Bundle K independent EDLA networks for K-dimensional output.
    Each network has the same hyperparameters, activation functions, and loss function.
    The output size is set to 1 for each network.
    However, each network can have its own parameters.
    input_size:     Number of input features
    hidden_size:    Number of hidden neurons
    hidden_layers:  Number of hidden layers
    output_size:    Number of output neurons (1 for each network)
    activation_fn:  Activation function for hidden layers
    last_activation_fn: Activation function for output layer
    loss_fc:       Loss function
    learning_rate: Learning rate for weight updates    
    reduction:     Reduction method for weight updates ('mean' or 'sum')     
    """

    def __init__(self,input_size, hidden_size, output_size, hidden_layers = 0, activation_fn = "Sigmoid", last_activation_fn = "Sigmoid", loss_fc="MSE", learning_rate=0.1, reduction='mean'):
        super(EDLA_Multi, self).__init__()

        self.input_size  = input_size
        self.output_size = output_size

        # Create as K EDLA networks
        self.edlas = nn.ModuleList()
        for i in range(output_size):
            self.edlas.append(
                EDLA(input_size=input_size, hidden_size = hidden_size, output_size=1, hidden_layers=hidden_layers, activation_fn=activation_fn, last_activation_fn=last_activation_fn, loss_fc=loss_fc,learning_rate=learning_rate,reduction=reduction))

    # -- Forward pass --
    def forward(self, x):
        with torch.no_grad():
            batch_size = x.shape[0]
            outputs = torch.zeros((batch_size, self.output_size)).to(x.device)
            for n, edla in enumerate(self.edlas):
                o, _ = edla(x)
                outputs[:,n] = o.squeeze()

        return outputs

    # -- Update weights of all EDLA networks --
    def update(self, input_data, target):    
        with torch.no_grad():
            for n, edla in enumerate(self.edlas):
                edla.update(model=edla, input_data=input_data, target=target[:, n].view(-1,1))


if __name__ == "__main__":
    import time
    # Example: Train XOR using EDLA_Multi
    # Choose device: 'cpu', 'cuda', or 'mps' (for Mac M1/M2)
    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #else:
    #    device = torch.device("cpu")
    
    # For testing, we will use CPU
    # Computation on MPS is slower than CPU.
    device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # XOR dataset
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32).to(device)
    Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32).to(device)

    # Model: 2 input, 4 hidden, 1 output, 1 hidden layer
    model = EDLA_Multi(input_size=2, hidden_size=4, output_size=1, hidden_layers=1, activation_fn="Sigmoid", last_activation_fn="Sigmoid", loss_fc="MSE", learning_rate=1.0).to(device)

    epochs = 2000

    # Training loop
    print("Training EDLA_Multi on XOR dataset...")
    start_time = time.time()
    for epoch in range(epochs):
        model.update(X, Y)
        if (epoch+1) % 200 == 0 or epoch == 0:
            out = model(X)
            loss = torch.mean((out - Y) ** 2).item()
            print(f"Epoch {epoch+1:4d} | Loss: {loss:.4f}")

    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    # Final predictions
    out = model(X)
    print("\nXOR predictions:")
    for x, y_pred in zip(X, out):
        print(f"Input: {x.cpu().numpy()}  Pred: {y_pred.item():.3f}")