import torch
from typing import List, Tuple
from torch import nn

class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
    
    def forward(self, input):
        """
            :param input: [bsz, in_features]
            :return result [bsz, out_features]
        """
        return torch.matmul(input, self.weight.t()) + self.bias


class MLP(torch.nn.Module):
    # 20 points
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super(MLP, self).__init__() 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)
        
        # Initializaton
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
    
    def _build_layers(self, input_size: int, 
                        hidden_sizes: List[int], 
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be aware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """
        hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            hidden_layers.append(Linear(input_size, hidden_sizes[i]))
            input_size = hidden_sizes[i]
        
        output_layer = Linear(hidden_sizes[-1], num_classes)
        
        return hidden_layers, output_layer

    def stable_sigmoid(self, x):
        threshold_high = 20.0
        threshold_low = -20.0

        result = torch.empty_like(x)

        result = torch.where(x > threshold_high, torch.tensor(1.0, dtype=x.dtype, device=x.device), result)
        result = torch.where(x < threshold_low, torch.tensor(0.0, dtype=x.dtype, device=x.device), result)
        result = torch.where((x <= threshold_high) & (x >= threshold_low), 1 / (1 + torch.exp(-x)), result)

        return result

    def stable_tanh(self, x):
        threshold_high = 20.0
        threshold_low = -20.0

        result = torch.empty_like(x)

        result = torch.where(x > threshold_high, torch.tensor(1.0, dtype=x.dtype, device=x.device), result)
        result = torch.where(x < threshold_low, torch.tensor(-1.0, dtype=x.dtype, device=x.device), result)
        result = torch.where((x <= threshold_high) & (x >= threshold_low), (torch.exp(2*x)-1)/(torch.exp(2*x)+1), result)

        return result
    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        
        if activation == 'relu':
            output = torch.maximum(inputs, torch.zeros_like(inputs))
        elif activation == 'tanh':
            # output = (torch.exp(inputs) - torch.exp(-inputs))/(torch.exp(inputs) + torch.exp(-inputs))
            output = self.stable_tanh(inputs)
        elif activation == 'sigmoid':
            # output = 1/(1 + torch.exp(-inputs))
            output = self.stable_sigmoid(inputs)
        else:
            raise NotImplementedError
        
        return output
        
    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        """ For bias set to zeros. For weights set to glorot normal """
        
        # Glorot initialization sqrt(6)/sqrt(n_in + n_out) in linear we have weights defined as n_out x n_in
        weight_range = torch.sqrt(torch.tensor(6.0))/torch.sqrt(torch.tensor(module.weight.shape[1]) + torch.tensor(module.weight.shape[0]))
        nn.init.uniform_(module.weight, -weight_range, weight_range)
        
        # Biases should be 0, but to ensure:
        nn.init.zeros_(module.bias)
        
        return
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first flattened to vectors. 
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.
        
        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """
        x = images.view(images.shape[0], -1)
        
        for layer in self.hidden_layers:
            h = layer(x)
            a = self.activation_fn(self.activation, h)
            x = a
            
        logits = self.output_layer(x)
        
        return logits
