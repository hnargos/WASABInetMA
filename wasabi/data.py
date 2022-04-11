# Standard library imports
import pickle
from typing import Union, Tuple
import os

# Third party imports
from sklearn.model_selection import train_test_split
from numpy.random import default_rng
from numpy import zeros as nzeros, corrcoef as ncorrcoef
import tqdm

import torch as t
from torch.utils.data import DataLoader, Dataset
from torch import load as tload
from torch import device, save, cat, mean, abs, zeros, tensor, zeros_like, randperm
import torch.cuda as cuda
from torch import Tensor
from torch import float32 as pytFl32
from .auxiliary_functions import load_config_from_model, load_config, make_dir


class DatasetXY(Dataset):
    """Class that defines a dataset for supervised learning."""

    def __init__(self, x: Tensor, y: Tensor, x_no_noise: Tensor = None, y_no_noise_all: Tensor = None):
        """
        :param x: input data
        :param y: target data
        :param x_no_noise: input data without noise
        :param y_no_noise_all: target data without noise and with all parameters in order {'dB0', 'B1', 'T1', 'T2'}. Should not be normed. (only relevant for GNLLonSigma_PhysM_alt_useTgts loss function atm)
        """
        self.x = x
        self.y = y

        if x_no_noise is None:
            self.x_no_noise = x.clone()
        else:
            self.x_no_noise = x_no_noise.to(device=x.device)

        if y_no_noise_all is None:
            self.y_no_noise_all = y.clone()
        else:
            self.y_no_noise_all = y_no_noise_all.to(device=y.device)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.x_no_noise[idx], self.y_no_noise_all[idx]

    def __len__(self):
        return self.x.shape[0]


class Data:
    """Class to load, save and store the feature and input data."""

    def __init__(self):
        """Class to load, save and store the feature and input data."""
        self.x_filepath = ''  # loading filepath
        self.y_filepath = ''  # loading filepath
        self.dev = None  # gpu device
        self.x_train = t.empty(0)  # split off feature trainings tensor
        self.x_val = t.empty(0)  # split off feature test tensor
        self.x_test = t.empty(0)  # split of feature validation tensor, always 5%
        self.y_train = t.empty(0)  # split off target trainings tensor
        self.y_val = t.empty(0)  # split off target test tensor
        self.y_test = t.empty(0)  # split of target validation tensor, always 5%
        self.train_loader = None  # DataLoader for the trainings data
        self.val_loader = None  # DataLoader for the test data
        self.test_loader = None  # DataLoader for the test data
        self._evaluation = False  # if true load the dataset only for evaluation
        self.raw_data = None  # if additional data is loaded it is placed into this variable
        self.x_no_noise = None  # contains the x_input data before noise gets added
        self.x_no_noise_train = None
        self.x_no_noise_val = None
        self.x_no_noise_test = None
        self.y_no_noise_all = None   # contains all 4 possible target parameters in order without noise
        self.y_no_noise_all_train = None  # contains all 4 possible target parameters in order without noise
        self.y_no_noise_all_val = None  # contains all 4 possible target parameters in order without noise
        self.y_no_noise_all_test = None  # contains all 4 possible target parameters in order without noise
        self.n_tgt_params = None  # get number of target parameters without the uncertainty
        self.config = None  # to store the configuration file should it be loaded
        self.norm_tgts_bounds = None  # stores the boundaries of the targets from norming
        self.norm_scale = None  # stores a rescaling factor for the norming range onto which the data gets projected
        self.norm_offset = None  # stores a offset factor for the norming range onto which the data gets projected (gets added)

    def _prep_data(self, data: Tensor) -> Tensor:
        """Standardize the data, assuming gaussian distribution.
        Takes the x_data and sets the mean to zero and the std to one.
        :param data: pytorch tensor with the trainings data.
        """
        # mean to zero
        data -= data.mean()
        # standard deviation to one
        data /= data.std()
        return data

    def _check_cuda(self, printing: bool = False):
        """Check if cuda is available and set self.dev.
        :param printing: if True will print the cuda device name."""
        if cuda.is_available():
            if printing:
                print(cuda.get_device_name(0))
            self.dev = device('cuda:0')
        else:
            self.dev = device('cpu')

    def add_gamma_std_noise(self, x: Tensor, y: Tensor, scale: float = 1., phantom=False) -> (Tensor, Tensor):
        """Adds noise based on a normal distribution whos standard deviation gets sampled by a gamma distribution.
        :param x: input tensor
        :param y: output tensor
        :param scale: gets multiplied with the noise term before that gets added
        :param phantom: if True will only apply noise once.
        :return: input tensor with 10 different noises and output tensor
        """
        # initialize random generator
        rng = default_rng()

        # lambda function for the std
        def std_dev():
            return rng.gamma(2, 0.005, 1)

        # get tensor dimensions
        tensor_len = x.shape[0]
        tensor_offsets = x.shape[1]

        # prep the no noise tensor
        x_no_noise = abs(x.detach().clone())
        x_no_noise_list = []
        y_no_noise_all = abs(y.detach().clone())
        y_no_noise_all_list = []

        # make x to numpy
        x = x.detach().numpy()

        # define output
        x_output = nzeros((int(tensor_len * 10), tensor_offsets))
        y_list = []

        # phantom=True => only add noise once, no multiplying data
        if phantom:
            x_output = nzeros((int(tensor_len), tensor_offsets))

            for i in range(tensor_len):
                x_output[i] = x[i] + rng.normal(0, std_dev(), tensor_offsets) * scale

            return tensor(x_output, dtype=pytFl32).to(self.dev), y.clone().detach().float().to(self.dev)

        # create loop iterator
        loop_iterator = tqdm.tqdm(range(10), ascii=True, position=0)

        # loop 10 times to get 10 different noise level for each spectrum
        for k in loop_iterator:
            for i in range(tensor_len):
                x_output[k * tensor_len + i] = x[i] + rng.normal(0, std_dev(), tensor_offsets) * scale
            y_list.append(y.clone().detach())
            x_no_noise_list.append(x_no_noise.clone().detach())
            y_no_noise_all_list.append(y_no_noise_all.clone().detach())
            loop_iterator.set_postfix([('adding noise ', ' gamma type')])  # update the progress bar

        # concatenate the y_list together and make a tensor
        y_output = cat(y_list).float()
        x_output = tensor(x_output, dtype=pytFl32)
        self.x_no_noise = cat(x_no_noise_list).float().to(self.dev)
        self.y_no_noise_all = cat(y_no_noise_all_list).float().to(self.dev)

        return x_output, y_output

    def _transf_target_vector_to_params(self, y_raw: Tensor, config: dict) -> Tensor:
        """Transforms the raw target vector into a target vector that only contains the parameters
        specified in the config for the neural net.

        :param y_raw: full target vector with all four parameters dB0, B1, T1, T2
        :param config: config file for the neural net
        :return: the transformed target vector
        """
        # only use the target parameters we want and send to device
        y = zeros((y_raw.shape[0], len(config.get('TYPE_PARAMS'))), dtype=pytFl32).to(self.dev)
        for i in range(len(config.get('TYPE_PARAMS'))):
            if config.get('TYPE_PARAMS')[i] == 'dB0':
                y[:, i] = y_raw[:, 0]
            elif config.get('TYPE_PARAMS')[i] == 'B1':
                y[:, i] = y_raw[:, 1]
            elif config.get('TYPE_PARAMS')[i] == 'T1':
                y[:, i] = y_raw[:, 2]
            elif config.get('TYPE_PARAMS')[i] == 'T2':
                y[:, i] = y_raw[:, 3]
            else:
                raise NameError('{} is not dB0, B1, T1 or T2. This function '
                                'only exists for these values'.format(config.get('TYPE_PARAMS')[i]))
        return y

    def save_BMCTool_data(self,
                          x_filepath: str = None,
                          y_filepath: str = None):
        """
        Saves the BMCTool data in the state it currently has in the object.
        If no file paths are given it assumes the filepaths in the config have a .pt ending.

        :param x_filepath: default the same filepath from where the x data gets loaded with '_w_noise.pt' added.
        :param y_filepath: default the same filepath from where the y data gets loaded with '_w_noise.pt' added.
        """
        # retrieve separate raw data
        x, y_raw = self.raw_data

        # save x
        if x_filepath is None:
            if self.x_filepath[-3:] != '.pt':
                print("The input x file was not a .pt file so there might be errors in saving the file with noise.")
            make_dir(os.path.split(self.x_filepath)[0])
            save(x, self.x_filepath[:-3] + '_w_noise.pt')
        else:
            make_dir(os.path.split(x_filepath)[0])
            save(x, x_filepath)

        # save y
        if y_filepath is None:
            if self.y_filepath[-3:] != '.pt':
                print("The input x file was not a .pt file so there might be errors in saving the file with noise.")
            make_dir(os.path.split(self.y_filepath)[0])
            save(x, self.y_filepath[:-3] + '_w_noise.pt')
        else:
            make_dir(os.path.split(y_filepath)[0])
            save(x, y_filepath)

    def _split_data(self,
                    x: Tensor,
                    y: Tensor,
                    evaluation: bool,
                    val_size: float,
                    test_size: float = 0.05,
                    pre_shuffle_dataset: bool = False):
        """
        Splits the data into training, validation and testing dataset. The test data set is always set
        to 5% unless it is a pure evaluation dataset.

        :param x: raw x data
        :param y: raw y data
        :param evaluation: If true put all the data in the testing set.
        :param val_size: defined in the config, sets the size of the validation set for hyperparameter optimization
        :param test_size: 0.05
        :param pre_shuffle_dataset: if true shuffles the data irrecoverably before splitting
        """
        # shuffles the whole dataset
        if pre_shuffle_dataset:
            index = randperm(x.shape[0])
            x = x[index]
            y = y[index]

            # if physical model is used
            if self.x_no_noise is not None:
                self.x_no_noise = self.x_no_noise[index]

        if not evaluation:
            # calculate the size of each set
            val_len = int(x.shape[0] * val_size)
            test_len = int(x.shape[0] * test_size)
            train_len = int(x.shape[0] - val_len - test_len)

            # split in train and test data
            self.x_train, self.y_train = x[:train_len].to(self.dev), y[:train_len].to(self.dev)
            self.x_val, self.y_val = x[train_len: - test_len].to(self.dev), y[train_len: - test_len].to(self.dev)
            self.x_test, self.y_test = x[- test_len:].to(self.dev), y[- test_len:].to(self.dev)

            # if physical model is used
            if self.x_no_noise is not None:
                self.x_no_noise_train = self.x_no_noise[:train_len].to(self.dev)
                self.x_no_noise_val = self.x_no_noise[train_len: - test_len].to(self.dev)
                self.x_no_noise_test = self.x_no_noise[- test_len:].to(self.dev)

            if self.y_no_noise_all is not None:
                self.y_no_noise_all_train = self.y_no_noise_all[:train_len].to(self.dev)
                self.y_no_noise_all_val = self.y_no_noise_all[train_len: - test_len].to(self.dev)
                self.y_no_noise_all_test = self.y_no_noise_all[- test_len:].to(self.dev)

        else:
            # split in train and test data for pure evaluation
            self.x_train, self.y_train = t.empty(0, device=self.dev), t.empty(0, device=self.dev)
            self.x_test, self.y_test = x.to(self.dev), y.to(self.dev)

            # if physical model is used
            if self.x_no_noise is not None:
                self.x_no_noise_train = t.empty(0, device=self.dev)
                self.x_no_noise_test = self.x_no_noise.to(self.dev)

            if self.y_no_noise_all is not None:
                self.y_no_noise_all_train = t.empty(0, device=self.dev)
                self.y_no_noise_all_test = self.y_no_noise_all.to(self.dev)

    def _set_config(self, config: Union[str, dict]):
        """
        Sets the self.config variable based on what type the given config is.
        If it is a dictionary it sets it directly if it is a string it expects that to be the filepath to a model or
        yaml file.

        :param config: str to a model or yaml file, or a dict of the config
        :return:
        """
        # load config file
        if isinstance(config, str):
            if config.split('.')[-1] == 'pt':
                self.config = load_config_from_model(config)
            elif config.split('.')[-1] == 'yaml':
                self.config = load_config(config)
            else:
                raise ValueError('Could not recognize file ending of the config. It has to be .yaml for a config file '
                                 'or .pt if you are loading from a model. '
                                 'The filepath was {}'.format(config))
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError('The config keyword has to be either filled with a dictionary or a string.')

        # check for validity of config values
        if self.config['LOSS_FKT'] == 'GNLL':
            assert len(self.config.get('TYPE_PARAMS',
                                       ['dB0', 'B1', 'T1', 'T2'])) * 2 == self.config['N_NEURONS'][-1], '' \
                            'The number of given paramters and the number of neurons in the output layer do not fit ' \
                            'to one another!'
        elif self.config['LOSS_FKT'] == 'MSELoss':
            assert len(self.config.get('TYPE_PARAMS',
                                       ['dB0', 'B1', 'T1', 'T2'])) == self.config['N_NEURONS'][-1], '' \
                            'The number of given paramters and the number of neurons in the output layer do not fit ' \
                            'to one another!'

    def _norm_tgts(self, norm_tgt_space: dict, y_raw: Tensor, norm_range: Union[int, list] = 1.) -> Tensor:
        """Norm targets based on norm_tgt_space to [0,1]. If config['NORM_RANGE'] it will norm to that range.

        Example: value of 2. with norm_tgt_space = [0., 4.]: value becomes 0.5
        Exanmple 2: value of 2. with norm_tgt_space = [0., 4.] and norm_range = 2: value becomes 1
        Example 2: value of 2. with norm_tgt_space = [0., 4.] and norm_range = [1., 2.]: value becomes 1.5

        :param norm_tgt_space: contains the original space boundaries
        :param y_raw: the data to be normed
        :param norm_range: The range to which it will normally be normed. int: [0,1] * norm_range; list: norm_range. Default = 1.
        """
        norm_offset = 0.  # if the norming range has to be shifted from lower boundary = 0

        # set unusual norming target ranges
        if isinstance(norm_range, int) or isinstance(norm_range, float):
            norm_scale = float(norm_range)
        elif isinstance(norm_range, list):
            assert len(norm_range) == 2, "The NORM_RANGE parameter has to consist of only the left and right " \
                                         "boundary written in a list. e.g. NORM_RANGE: [0., 1.]"

            norm_scale = float(norm_range[1]) - float(norm_range[0])
            norm_offset = float(norm_range[0])
        else:
            raise TypeError("The data type of NORM_RANGE hast to be int or list, it is however: {}".format(type(norm_range)))

        # norm dB0
        if norm_tgt_space.get('dB0', False):
            bounds = norm_tgt_space.get('dB0')
            y_raw[:, 0] = (y_raw[:, 0] - bounds[0]) / (bounds[1] - bounds[0]) * norm_scale + norm_offset
        # norm B1
        if norm_tgt_space.get('B1', False):
            bounds = norm_tgt_space.get('B1')
            y_raw[:, 1] = (y_raw[:, 1] - bounds[0]) / (bounds[1] - bounds[0]) * norm_scale + norm_offset
        # norm T1
        if norm_tgt_space.get('T1', False):
            bounds = norm_tgt_space.get('T1')
            y_raw[:, 2] = (y_raw[:, 2] - bounds[0]) / (bounds[1] - bounds[0]) * norm_scale + norm_offset
        # norm T2
        if norm_tgt_space.get('T2', False):
            bounds = norm_tgt_space.get('T2')
            y_raw[:, 3] = (y_raw[:, 3] - bounds[0]) / (bounds[1] - bounds[0]) * norm_scale + norm_offset

        self.norm_scale = norm_scale
        self.norm_offset = norm_offset

        return y_raw

    def undo_norm_tgts(self, predictions: Tensor,
                       prediction_tgts: Tensor,
                       both_toggle: bool = True,
                       all_toggle: bool = False) -> (Tensor, Tensor):
        """Undoes the norm for the predictions and predictions_targets based on NORM_TGTs and NORM_RANGE.

        :param prediction_tgts: The target data for the predictions. They do not contain uncertainties.
        :param predictions: the normed data in tensor form [number of predictions, parameter and uncertainties]
        :param both_toggle: if set to false calculates only the prediction tensor without uncertainties
        :param all_toggle: if True will assume full parameter range of ['dB0', 'B1', 'T1', 'T2']
        """
        # get predictions order
        type_params = self.config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2'])
        if all_toggle:
            type_params = ['dB0', 'B1', 'T1', 'T2']

        for j, key in enumerate(type_params):
            if self.norm_tgts_bounds.get(key, False):
                bounds = self.norm_tgts_bounds.get(key)
                predictions[:, j] = (predictions[:, j] - self.norm_offset) / self.norm_scale * \
                                    (bounds[1] - bounds[0]) + bounds[0]
                if both_toggle:
                    if predictions.shape[1] > prediction_tgts.shape[1]:
                        predictions[:, j + self.n_tgt_params] = predictions[:, j + self.n_tgt_params] / self.norm_scale\
                                                                * (bounds[1] - bounds[0])
                    prediction_tgts[:, j] = (prediction_tgts[:, j] - self.norm_offset) / self.norm_scale * \
                                            (bounds[1] - bounds[0]) + bounds[0]

        return predictions, prediction_tgts

    def _check_for_norming_tgts(self, norm_tgt_space: Union[bool, dict],
                                y_raw: Tensor,
                                norm_range: Union[int, list] = 1.) -> Tensor:
        """Checks if the targets have to be normed.

        :param norm_tgt_space: if true to the maximal data spread +- 0.1 for every parameter
        If it is a dict then it uses the dict
        :param y_raw: the data to be normed
        :param norm_range: The range to which it will normally be normed. int: [0,1] * norm_range; list: norm_range. Default = 1.
        """
        self.norm_tgts_bounds = norm_tgt_space

        # check for bool and thus default values
        if isinstance(norm_tgt_space, bool):
            if norm_tgt_space:
                # norm to default +- 0.1
                self.norm_tgts_bounds = {'dB0': [y_raw[:, 0].min() - 0.1, y_raw[:, 0].max() + 0.1],
                                         'B1': [y_raw[:, 1].min() - 0.1, y_raw[:, 1].max() + 0.1],
                                         'T1': [y_raw[:, 2].min() - 0.1, y_raw[:, 2].max() + 0.1],
                                         'T2': [y_raw[:, 3].min() - 0.1, y_raw[:, 3].max() + 0.1]}
                self.config['NORM_TGTS'] = self.norm_tgts_bounds
                return self._norm_tgts(self.norm_tgts_bounds, y_raw, norm_range)
            else:
                return y_raw

        # check for dictionary
        elif isinstance(norm_tgt_space, dict):
            # norm to given boundaries
            return self._norm_tgts(norm_tgt_space, y_raw, norm_range)

        # give TypeError
        else:
            raise TypeError('The defined norm_tgt_space or NORM_TGTS is neither a dictionary nor a boolean.')

    def _set_loader(self,
                    x: Tensor = None,
                    y: Tensor = None,
                    x_no_noise: Tensor = None,
                    y_no_noise_all: Tensor = None,
                    kwargs: dict = None,
                    loader_type: str = None) -> DataLoader:
        """This function creates a DataLoader for the given data.\n
        IMPORTANT: If self._evaluation is True, train and val datasets can not be set with this function!

        :param x: input data, if not set, it will use self.x_(train/val/test) based on loader_type
        :param y: target data, if not set, it will use self.y_(train/val/test) based on loader_type
        :param x_no_noise: input data without noise, if not set the data loader will return zeros as the last element
        :param y_no_noise_all: target data without noise and with all four parameters in the order {'dB0', 'B1', 'T1', 'T2'}. Only relevant for GNLLonSigma_phys_m_alt_useTgts atm.
        :param kwargs: kwargs of the DataLoader class of pytorch
        :param loader_type: If this is not set it returns a DataLoader based on x,y. Can be set to 'train', 'val' or
        'test' and specifies which loader to set.
        :return: Optional, returns only if loader_type = None a DataLoader based on (x,y)
        """
        # returns unspecified DataLoader
        if loader_type is None:
            return DataLoader(DatasetXY(x, y, x_no_noise, y_no_noise_all), **kwargs)
        # sets self.train_loader
        elif loader_type == 'train':
            if not self._evaluation:
                if x is None:
                    x = self.x_train.to(self.dev)
                if y is None:
                    y = self.y_train.to(self.dev)
                if x_no_noise is None:
                    x_no_noise = self.x_no_noise_train
                if y_no_noise_all is None:
                    y_no_noise_all = self.y_no_noise_all_train
                self.train_loader = DataLoader(DatasetXY(x, y, x_no_noise=x_no_noise,
                                                         y_no_noise_all=y_no_noise_all), **kwargs)
            else:
                self.train_loader = None
        # sets self.val_loader
        elif loader_type == 'val':
            if not self._evaluation:
                if x is None:
                    x = self.x_val.to(self.dev)
                if y is None:
                    y = self.y_val.to(self.dev)
                if x_no_noise is None:
                    x_no_noise = self.x_no_noise_val
                if y_no_noise_all is None:
                    y_no_noise_all = self.y_no_noise_all_val
                self.val_loader = DataLoader(DatasetXY(x, y, x_no_noise=x_no_noise,
                                                       y_no_noise_all=y_no_noise_all), **kwargs)
            else:
                self.val_loader = None
        # sets self.test_loader
        elif loader_type == 'test':
            if x is None:
                x = self.x_test.to(self.dev)
            if y is None:
                y = self.y_test.to(self.dev)
                if x_no_noise is None:
                    x_no_noise = self.x_no_noise_test
                if y_no_noise_all is None:
                    y_no_noise_all = self.y_no_noise_all_test
            self.test_loader = DataLoader(DatasetXY(x, y, x_no_noise=x_no_noise,
                                                    y_no_noise_all=y_no_noise_all), **kwargs)
        else:
            raise Exception("Could not set a DataLoader. type(x) = {}, type(y) = {}, "
                            "kwargs = {}, loader_type = {}".format(type(x), type(y), kwargs, loader_type))

    def _change_train_loader_batchsize(self,
                                       batch_size: Union[int, dict],
                                       cur_epoch: int,
                                       kwargs: dict = None):
        """Changes the train loader to the given batch size, based on the current epoch.

        :param batch_size: from the config file
        :param cur_epoch: current epoch
        :param kwargs: default {'shuffle': True}; it is recommended for the train loader
        """
        if not self._evaluation:
            # need the kwargs initialized to hand it batch_size later
            if kwargs is None:
                kwargs = {'shuffle': True}

            if isinstance(batch_size, int):
                # when handed a integer
                kwargs['batch_size'] = batch_size
                self._set_loader(loader_type='train', kwargs=kwargs)
            elif isinstance(batch_size, dict):
                # when handed a dictionary, for example during reloading of the net
                cur_batch_size = 0
                for i in range(cur_epoch + 1):
                    if batch_size.get(i, False):
                        cur_batch_size = batch_size.get(i)
                kwargs['batch_size'] = cur_batch_size
                self._set_loader(loader_type='train', kwargs=kwargs)
            else:
                raise TypeError("The batch_size had the wrong datatype {}.".format(type(batch_size)))

    def load_data_tensor(self,
                         config: Union[str, dict],
                         x_file: Union[str, Tensor] = None,
                         y_file: Union[str, Tensor] = None,
                         printing: bool = True,
                         val_size: float = 0.25,
                         evaluation: bool = False,
                         add_noise: bool = True,
                         create_zeros_tgts: bool = False,
                         noise_scale: float = 1.
                         ):
        """
        :param add_noise: True to add noise to the data
        :param config: (str) then it expects a filepath to a model if the ending is .pt or if the ending is .yaml it expects a config file; (dict) then it expects a loaded config file in form of a dictionary
        :param evaluation: if true load the entire dataset only for evaluation
        :param val_size: portion of the data that is utilized for the test set
        :param x_file: filepath for the features or expects data in the form Tensor[n_samples, 31 offsets]
        :param y_file: filepath for the targets or expects data in the form Tensor[n_samples, m_targets]
        :param printing: True if the function should use print() for details
        :param create_zeros_tgts: This has to be true if there is no target data. This will not enable you to calculate a loss, only predictions. It creates a tensor with the same dimensions as x, however filled with zeros.
        :param noise_scale: Gets multiplied with the noise before it gets added to the x input.

        Loads feature and target dataset with torch.load() and splits them in
        trainings and testing sets.

        If possible it moves the resulting tensors to the gpu 'cuda:0'.

        Then it creates pytorch DataLoader Objects with that data. The order
        of the parameters is: dB0, B1, T1, T2, dB0 std, B1 std, T1 std, T2 std; unless otherwise specified.
        """
        # load config file
        self._set_config(config)

        self._evaluation = evaluation  # if true load the dataset only for evaluation
        val_size = self.config.get('DATA_SPLIT', val_size)  # set test set size

        # get filepath or assign data
        if x_file is not None:
            if isinstance(x_file, str):
                self.x_filepath = x_file
                x_file = None
            elif isinstance(x_file, Tensor):  # assign data
                x = x_file
            else:
                raise TypeError('x_file is of the wrong datatype, it has to be str, Tensor or not given (None).')
        else:
            self.x_filepath = self.config.get('DATA_X_PATH')
        if y_file is not None:
            if isinstance(y_file, str):
                self.y_filepath = y_file
                y_file = None
            elif isinstance(y_file, Tensor):  # assign data
                y_raw = y_file
            else:
                raise TypeError('y_file is of the wrong datatype, it has to be str, Tensor or not given (None).')
        else:
            self.y_filepath = self.config.get('DATA_Y_PATH')

        # load data if not assigned yet
        if x_file is None:
            x = tload(self.x_filepath)  # feature data
        if create_zeros_tgts:
            y_raw = zeros((len(x), len(self.config['TYPE_PARAMS'])), dtype=pytFl32)
        elif y_file is None:
            y_raw = tload(self.y_filepath)  # target data
        elif y_raw is not None:
            pass
        else:
            raise Exception('Something went wrong with the y data.')

        # adds noise if a noise type is given and noise is requested (default = True)
        if self.config.get('NOISE', 'no') == 'gamma_std' and add_noise:
            x, y_raw = self.add_gamma_std_noise(x, y_raw, scale=noise_scale)
        # fill self.y_no_noise_all with all possible target parameters
        else:
            self.y_no_noise_all = y_raw.clone()

        # Check for CUDA or use forced device
        if self.config.get('FORCE_CPUGPU') is not None:
            if self.config.get('FORCE_CPUGPU') == 'cpu':
                self.dev = device('cpu')
            elif self.config.get('FORCE_CPUGPU') == 'gpu':
                self.dev = device('cuda:0')
            elif not self.config.get('FORCE_CPUGPU') or self.config.get('FORCE_CPUGPU'):
                self._check_cuda(printing)
        else:
            self._check_cuda(printing)

        # take abs of z-spectrum
        x = abs(x)
        if self.x_no_noise is not None:
            self.x_no_noise = abs(self.x_no_noise)

        # possibly norm targets
        self.norm_tgts_bounds = self.config.get('NORM_TGTS', False)
        y_raw = self._check_for_norming_tgts(self.config.get('NORM_TGTS', False),
                                             y_raw,
                                             self.config.get('NORM_RANGE', 1.))

        # set raw data into variable
        self.raw_data = (x, y_raw)

        # send to device
        x = x.to(self.dev)

        # only use the target parameters we want and send to device
        if self.config.get('TYPE_PARAMS') is not None:
            y = self._transf_target_vector_to_params(y_raw, self.config).to(self.dev)
        else:
            y = y_raw.to(self.dev)

        if evaluation:
            pre_shuffle_dataset = False
        else:
            pre_shuffle_dataset = True

        # split in train and test data
        self._split_data(x, y, evaluation,
                         val_size=val_size,
                         pre_shuffle_dataset=pre_shuffle_dataset)

        # get number of target parameters without the uncertainty
        self.n_tgt_params = len(self.config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2']))

        if printing:
            print(f'X_train.shape = {self.x_train.shape} ; '
                  f'X_val.shape = {self.x_val.shape} ; '
                  f'X_test.shape = {self.x_test.shape}')
            print(f'y_train.shape = {self.y_train.shape} ; '
                  f'y_val.shape = {self.y_val.shape} ;  '
                  f'y_test.shape = {self.y_test.shape}')

        # create DataLoader for pytorch, train_loader gets created during training
        self._set_loader(loader_type='val', kwargs={'batch_size': 128})
        self._set_loader(loader_type='test', kwargs={'batch_size': 128})
        self.train_loader = None

    def load_phantom(self, filepath: str,
                     config: Union[str, dict],
                     printing: bool = False,
                     add_noise: bool = False,
                     noise_scale: float = 1.):
        """Loads a dictionary of the phantom data. Only loads data for evaluation.

        :type config: (str) then it expects a filepath to a model or a config file; (dict) then it expects a loaded config file
        :param filepath: the filepath to the phantom
        :param config: configuration file for the neural network
        :param printing: if true prints the cuda device name
        :param noise_scale: Gets multiplied with the noise before it gets added to the x input.
        :param add_noise: True to add noise to the data
        """
        # load data
        with open(filepath, 'rb') as file:
            self.raw_data = pickle.load(file)

        # load config
        self._set_config(config)

        # set flag for evaluation
        self._evaluation = True

        # Check for CUDA
        self._check_cuda(printing)

        # assign temporary variables
        matrix_size = len(self.raw_data['phantom']['t1'])  # get phantom size, assumes square image
        z_specs = self.raw_data['z_specs']  # get the z_spectra dictionary
        b0_shift = self.raw_data['phantom']['b0_shift']  # is matrix
        b1_inhom = self.raw_data['phantom']['b1_inhom']  # is matrix
        t1 = self.raw_data['phantom']['t1']  # is matrix
        t2 = self.raw_data['phantom']['t2']  # is matrix

        # create and fill the z spectra matrix with loaded values or zero spectra
        self.x_test = t.zeros((len(z_specs), 31), dtype=pytFl32)
        y_raw = t.zeros((len(z_specs), 4), dtype=pytFl32)

        k = 0  # counter for the flat predictions array
        for i in range(matrix_size):
            for j in range(matrix_size):
                if z_specs.get((i, j)) is not None:
                    # fill input tensor with z-spectra that exist
                    self.x_test[k] = tensor(z_specs.get((i, j)), dtype=pytFl32)

                    # fill target tensor with all possible data
                    y_raw[k] = tensor([b0_shift[i, j],
                                       b1_inhom[i, j],
                                       t1[i, j],
                                       t2[i, j]], dtype=pytFl32)
                    k += 1

        # adds noise if a noise type is given and noise is requested (default = True)
        if add_noise:
            self.x_test, y_raw = self.add_gamma_std_noise(self.x_test, y_raw, scale=noise_scale, phantom=True)

        # possibly norm targets
        self.norm_tgts_bounds = self.config.get('NORM_TGTS', False)
        y_raw = self._check_for_norming_tgts(self.config.get('NORM_TGTS', False),
                                             y_raw,
                                             self.config.get('NORM_RANGE', 1.))

        # choose only use the target parameters we want
        if self.config.get('TYPE_PARAMS') is not None:
            self.y_test = self._transf_target_vector_to_params(y_raw, self.config)
        else:
            self.y_test = y_raw

        # send to device
        self.x_test = self.x_test.to(self.dev)
        self.y_test = self.y_test.to(self.dev)

        # create test_loader
        self._set_loader(loader_type='test', kwargs={'batch_size': 64})

        # set the number of parameters for the neural net
        self.n_tgt_params = len(self.config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2']))
