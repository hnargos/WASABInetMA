# Standard library imports
from typing import Tuple, Any

# Third party imports

# own scripts


# build layers without kwargs
def build_layers(number_of_layers: int = 2,
                 layer_start=None,
                 layer_frag=None,
                 layer_end=None,
                 neurons_start=None,
                 neurons=None,
                 neurons_end=None,
                 ) -> Tuple[list, list]:
    """Build a layer with given specs

    :param number_of_layers: minimum 0, then the layer_start and layer_end gets applied.
    For every number above 0 one layer_frag gets applied in the middle.
    :param layer_start: The first element of the layer list, can be list with multiple elements.
    :param layer_frag: The fragment of the network that appears again and again, the number of layers
    refers to how often this gets added in the middle.
    :param layer_end: The last element of the layer list.
    :param neurons_start: number of input features
    :param neurons: number of neurons per layer
    :param neurons_end: number of output features
    :return: layers as list, n_neurons as list as it would be in the config
    """
    if neurons_end is None:
        neurons_end = [8]
    if neurons is None:
        neurons = [128]
    if neurons_start is None:
        neurons_start = [31]
    if layer_end is None:
        layer_end = ['linear']
    if layer_start is None:
        layer_start = []
    if layer_frag is None:
        layer_frag = ['linear', 'elu']

    assert number_of_layers >= 0, "The number of layers has to be at least zero."
    layers = [] + layer_start
    n_neurons = [] + neurons_start

    for i in range(number_of_layers - 1):
        layers += layer_frag
        n_neurons += neurons

    layers += layer_end
    n_neurons += neurons_end

    return layers, n_neurons


# build layer grid
def build_layer_grid(number_of_layers: Any, **kwargs) -> Tuple[list, list]:
    """ Builds two lists with the layer specifications.

    :param number_of_layers: see build_layers, has to be possible to loop over, e.g. range(4), [0,1,2]
    :return: layer_grid list of lists, n_neurons_grid list of lists
    """
    layer_grid = []
    n_neurons_grid = []
    for i in number_of_layers:
        layers, n_neurons = build_layers(number_of_layers=i+1, **kwargs)
        layer_grid.append(layers)
        n_neurons_grid.append(n_neurons)
    return layer_grid, n_neurons_grid
