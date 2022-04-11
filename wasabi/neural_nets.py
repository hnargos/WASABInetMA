# Third party imports
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch import zeros_like


#### Linear Layers ####


def _add_identity(net: nn.Sequential(), index: str, **layer_kwargs):
    """Adds a identity layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.Identity(**layer_kwargs))


def _add_linear(net: nn.Sequential(), index: str, in_neuron: int, out_neuron: int, **layer_kwargs):
    """Adds a linear/dense layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param in_neuron: number of input neurons
    :param out_neuron: number of output neurons
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.Linear(in_neuron, out_neuron, **layer_kwargs))


#### Convolution Layers ####


def _add_conv1d(net: nn.Sequential(), index: str, in_neuron: int, out_neuron: int, **layer_kwargs):
    """Adds a Conv1d layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param in_neuron: number of input neurons
    :param out_neuron: number of output neurons
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.Conv1d(in_neuron, out_neuron, **layer_kwargs))


def _add_conv2d(net: nn.Sequential(), index: str, in_neuron: int, out_neuron: int, **layer_kwargs):
    """Adds a Conv2d layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param in_neuron: number of input neurons
    :param out_neuron: number of output neurons
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.Conv2d(in_neuron, out_neuron, **layer_kwargs))


#### Non-Linear Activations####


def _add_elu(net: nn.Sequential(), index: str, **layer_kwargs):
    """Adds a ELU activation layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.ELU(**layer_kwargs))


def _add_leakyrelu(net: nn.Sequential(), index: str, **layer_kwargs):
    """Adds a LeakyReLU layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.LeakyReLU(**layer_kwargs))


def _add_logsigmoid(net: nn.Sequential(), index: str, **layer_kwargs):
    """Adds a LogSigmoid activation layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.LogSigmoid(**layer_kwargs))


def _add_relu(net: nn.Sequential(), index: str, **layer_kwargs):
    """Adds a ReLU layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.ReLU(**layer_kwargs))


def _add_sigmoid(net: nn.Sequential(), index: str, **layer_kwargs):
    """Adds a Sigmoid activation layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.Sigmoid(**layer_kwargs))


def _add_softplus(net: nn.Sequential(), index: str, **layer_kwargs):
    """Adds a Softplus activation layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.Softplus(**layer_kwargs))


def _add_tanh(net: nn.Sequential(), index: str, **layer_kwargs):
    """Adds a Tanh activation layer.

    :param net: contains the neural net structure
    :param index: name of the layer
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    net.add_module(index, nn.Tanh(**layer_kwargs))


#### Custom split activation functions ####

class SoftplusSplit(nn.Module):
    """
    Basically the same as the softplus activation function except one can specify which neurons to apply to to.
    """
    def __init__(self, c_split: list, beta: int = 1, threshold: int = 20) -> None:
        """

        :param c_split: [n_start_neuron, m_end_neuron] onto this interval the function gets applied
        :param beta: see pytorch (https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html#torch.nn.Softplus)
        :param threshold: see pytorch (https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html#torch.nn.Softplus)
        """
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.c_split = c_split

    def forward(self, input: Tensor) -> Tensor:
        ret1 = F.softplus(input[self.c_split[0]: self.c_split[1]],
                          self.beta,
                          self.threshold)
        ret = zeros_like(input)
        ret[:self.c_split[0]] = input[:self.c_split[0]]
        ret[self.c_split[1]:] = input[self.c_split[1]:]
        ret[self.c_split[0]: self.c_split[1]] = ret1
        return ret


def _add_softplus_split(net: nn.Sequential(), index: str, **layer_kwargs):
    """

    :param net: contains the neural net structure
    :param index: name of the layer
    :param layer_kwargs: dict has to contain the parameter: 'c_split': [int n_neuron_start_split, int m_neuron_end_split] that specifies which neurons the layer applys to. + Can contain all the kwargs softplus could have (see pytorch documentation)
    """
    c_split = layer_kwargs['c_split']
    assert c_split[0] < c_split[1], "The later c_split integer must be larger then the first."

    net.add_module(index, SoftplusSplit(**layer_kwargs))


#### Dropout Layer ####


def _add_dropout(net: nn.Sequential(), index: str, **layer_kwargs):
    """Adds a Dropout layer

    :param net: contains the neural net structure
    :param index: name of the layer
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    :return:
    """
    net.add_module(index, nn.Dropout(**layer_kwargs))


def _create_modules(net: nn.Sequential(), layers: list, n_neurons: list, layer_kwargs: dict):
    """Creates the neural net structure.

    :param net: contains the neural net structure
    :param layers: types of layers, all in lowercase letters
    :param n_neurons: number of neurons per layer
    :param layer_kwargs: all the kwargs the type of layer could accept (see pytorch documentation)
    """
    j = 0  # layer neuron number, only increases when necessary
    # iterate over layers
    for i in range(len(layers)):
        # add Identity layer
        if layers[i] == 'identity':
            _add_identity(net, str(i), **layer_kwargs.get(i))
            j += 1
        # add Linear layer
        if layers[i] == 'linear':
            _add_linear(net, str(i), n_neurons[j], n_neurons[j + 1], **layer_kwargs.get(i, {}))
            j += 1
        # add Conv1d layer
        if layers[i] == 'conv1d':
            _add_conv1d(net, str(i), n_neurons[j], n_neurons[j + 1], **layer_kwargs.get(i, {}))
            j += 1
        # add Conv2d layer
        if layers[i] == 'conv2d':
            _add_conv2d(net, str(i), n_neurons[j], n_neurons[j + 1], **layer_kwargs.get(i, {}))
            j += 1

        #### Activations ####
        # add ELU activation
        elif layers[i] == 'elu':
            _add_elu(net, str(i), **layer_kwargs.get(i, {}))
        # add LeakyReLU activation
        elif layers[i] == 'leakyrelu':
            _add_leakyrelu(net, str(i), **layer_kwargs.get(i, {}))
        # add LogSigmoid activation
        elif layers[i] == 'logsigmoid':
            _add_logsigmoid(net, str(i), **layer_kwargs.get(i, {}))
        # add ReLU activation
        elif layers[i] == 'relu':
            _add_relu(net, str(i), **layer_kwargs.get(i, {}))
        # add Sigmoid activation
        elif layers[i] == 'sigmoid':
            _add_sigmoid(net, str(i), **layer_kwargs.get(i, {}))
        # add Softplus activation
        elif layers[i] == 'softplus':
            _add_softplus(net, str(i), **layer_kwargs.get(i, {}))
        # add Tanh activation
        elif layers[i] == 'tanh':
            _add_tanh(net, str(i), **layer_kwargs.get(i, {}))
        elif layers[i] == 'softplussplit':
            _add_softplus_split(net, str(i), **layer_kwargs.get(i, {}))

        #### Dropout Layer ####
        # add a Dropout Layer
        elif layers[i] == 'dropout':
            _add_dropout(net, str(i), **layer_kwargs.get(i, {}))


class NeuralNet(nn.Module):
    def __init__(self, layers: list, n_neurons: list, layer_kwargs):
        """The neural net gets constructed upon initialization."""
        super().__init__()
        if layer_kwargs is None:
            layer_kwargs = {}

        self.net = nn.Sequential()  # create empty net
        _create_modules(self.net, layers, n_neurons, layer_kwargs)  # fill empty net based on config file

    def forward(self, x):
        """Sends one mini batch forward through the net and returns the result.
        :param x: a mini batch
        """
        return self.net(x)


# pytorch neural net class
class WASABI_net2(nn.Module):
    """Creates a pytorch neural net with the specifications:

    self.net = nn.Sequential(nn.Linear(31, 128),
                            nn.Tanh(),\n
                            nn.Linear(128, 64),\n
                            nn.Tanh(),\n
                            nn.Linear(64, 3))
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(31, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 3))

    def forward(self, x):
        """Sends one mini batch forward through the net and returns the result."""
        return self.net(x)


# pytorch neural net class
class DeepCEST_net(nn.Module):
    """Creates a pytorch neural net with the specifications:

    self.net = nn.Sequential(nn.Linear(31, 100),
                             nn.ELU(), \n
                             nn.Linear(100, 100), \n
                             nn.ELU(), \n
                             nn.Linear(100, 100), \n
                             nn.ELU(), \n
                             nn.Linear(100, 3))
    """

    def __init__(self):
        super().__init__()
        print('OLD NET not recommended for use')

        self.net = nn.Sequential(nn.Linear(31, 100),
                                 nn.ELU(),
                                 nn.Linear(100, 100),
                                 nn.ELU(),
                                 nn.Linear(100, 100),
                                 nn.ELU(),
                                 nn.Linear(100, 3))

    def forward(self, x):
        """Sends one mini batch forward through the net and returns the result."""
        return self.net(x)


# pytorch neural net class
class DeepCEST_real_net(nn.Module):
    """Creates a pytorch neural net with the specifications:

    self.net = nn.Sequential(nn.Linear(31, 100),
                             nn.ELU(), \n
                             nn.Linear(100, 100), \n
                             nn.ELU(), \n
                             nn.Linear(100, 100), \n
                             nn.ELU(), \n
                             nn.Linear(100, 3))
    """

    def __init__(self):
        super().__init__()
        print('OLD NET not recommended for use')

        self.net = nn.Sequential(nn.Linear(31, 100),
                                 nn.ELU(),
                                 nn.Linear(100, 100),
                                 nn.ELU(),
                                 nn.Linear(100, 100),
                                 nn.ELU(),
                                 nn.Linear(100, 6))

    def forward(self, x):
        """Sends one mini batch forward through the net and returns the result."""
        return self.net(x)


class WASABI_net(nn.Module):
    """Creates a pytorch neural net with the specifications:

    self.net = nn.Sequential(
                nn.Linear(31,128),\n
                nn.Tanh(),\n
                nn.Linear(128,3))
    """

    def __init__(self):
        super().__init__()
        print('OLD NET not recommended for use')

        self.net = nn.Sequential(nn.Linear(31, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 3))

    def forward(self, x):
        """Sends one mini batch forward through the net and returns the result."""
        return self.net(x)
