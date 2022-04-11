# Standard library imports
from typing import Optional, Tuple, Union
import tqdm
import os
import zipfile
import shutil

# third party imports
import yaml
from torch import load, Tensor, mean, exp, abs, cos, sqrt, transpose, pow, sin, tensor, sign
from math import pi


def wasabiti_new(x: Tensor,
                 trec: Tensor,
                 b0_shift: Tensor = 0.,
                 b1: Tensor = 1.,
                 t1: Tensor = 2.,
                 t2: Tensor = 0.1,
                 tp: Tensor = 0.005,
                 gamma: Tensor = 42.5764,
                 ret_shape: Tensor = None) -> Tensor:
    """
    Full Analytical WASABI approximation depending on b0_shift, B1, T1 and T2
    see: https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1118/1.597063

    :param x: frequency offsets [ppm]
    :param trec: recover time between different offsets/measurements. Can be a float or an array [s]
    :param b0_shift: b0_shift-shift [ppm]
    :param b1: relative B1 value []
    :param t1: longitudinal relaxation time (T_1) [s]
    :param t2: transversal relaxation time (T_2) [s]
    :param tp: duration of the WASABI pulse [s]
    :param gamma: gyromagnetic ratio [MHz/T]
    :param ret_shape: matrix filled with ones to be used as in-between to get the correct tensor multiplications
    """

    athird = 1./3.
    w1 = gamma * 2. * pi * ret_shape * b1[:, None] * 3.75
    da = (x + ret_shape * b0_shift[:, None]) * 3 * gamma * pi * 2.
    R1 = 1. / (ret_shape * t1[:, None])
    R2 = 1. / (ret_shape * t2[:, None])

    Mzi = 1. - exp(-R1 * trec)

    p = 2. * R2 + R1
    q = R2 ** 2. + 2. * R1 * R2 + da ** 2. + w1 ** 2.
    r = R1 * R2 ** 2. + R2 * w1 ** 2. + da ** 2. * R1

    a = (3. * q - p ** 2.) / 3.
    b = (2. * p ** 3. - 9. * p * q + 27. * r) / 27.
    c = b ** 2. / 4. + a ** 3. / 27.

    # replaced cube root to enable negative results
    sqrtc = sqrt(c)
    temp1 = abs(-b / 2. + sqrtc)
    temp2 = abs(-b / 2. - sqrtc)
    A = temp1.sign() * pow(temp1, athird)
    B = temp2.sign() * pow(temp2, athird)

    a1 = -(p / 3.) + A + B
    a2i = 1. / 2. * sqrt(tensor(3.)) * (A - B)
    a2r = -(p / 3.) - (A + B) / 2.
    a2 = a2r + 1j * a2i

    m1 = (((R2 + a1) ** 2. + da ** 2.) * (R1 + Mzi * a1)) / (a1 * (a2i ** 2. + (a1 - a2r) ** 2.))
    m2 = (((R2 + a2) ** 2. + da ** 2.) * (R1 + Mzi * a2)) / (a2 * (a2 - a1) * (2j * a2i))
    mss = (R1 * (da ** 2. + R2 ** 2.)) / (R1 * (da ** 2. + R2 ** 2.) + R2 * w1 ** 2.)

    m = mss + m1 * exp(a1 * tp) + 2 * exp(a2r * tp) * (m2.real * cos(a2i * tp) - m2.imag * sin(a2i * tp))
    m = m + (1 - m) * (1 - exp(-R1 * 0.00553))  # relaxation during dead_time (30 µs) and spoiler (5.5 ms)

    return abs(m.float())


def wasabiti_alt(x: Tensor,
                 trec: Tensor,
                 b0_shift: Tensor = 0.,
                 b1: Tensor = 1.0,
                 t1: Tensor = 2.,
                 t2: Tensor = 0.1,
                 freq: Tensor = 127.7292,
                 tp: Tensor = 0.005,
                 gamma: Tensor = 42.5764,
                 ret_shape: Tensor = None
                 ) -> Tensor:
    """Full Analytical WASABI approximation depending on b0_shift, B1, T1 and T2
    including relaxation during dead_time (30 µs) and spoiler (5.5 ms)
    see: https://github.com/schuenke/WANTED/blob/main/fit/fit_functions.py

    :param x: frequency offsets [ppm]
    :param trec: recover time between different offsets/measurements. Can be a float or an array [s]
    :param b0_shift: b0_shift-shift [ppm]
    :param b1: relative B1 value []
    :param t1: longitudinal relaxation time (T_1) [s]
    :param t2: transversal relaxation time (T_2) [s]
    :param freq: frequency of the NMR system [MHz] (e.g. 128 for a 3T scanner)
    :param tp: duration of the WASABI pulse [s]
    :param gamma: gyromagnetic ratio [MHz/T]
    :param ret_shape: matrix filled with ones to be used as in-between to get the correct tensor multiplications
    """
    da = (x + ret_shape * b0_shift[:, None]) * freq * 2 * pi
    w1 = gamma * ret_shape * b1[:, None] * 2 * pi * 3.75
    dawi = da ** 2 + w1 ** 2

    r1 = 1. / (ret_shape * t1[:, None])
    r2 = 1. / (ret_shape * t2[:, None])
    r1p = (r1 * da ** 2) / (da ** 2 + w1 ** 2) + (r2 * w1 ** 2) / (da ** 2 + w1 ** 2)
    r2p = (2. * r2 + r1) / 2. - r1p / 2.  # according to https://doi.org/10.1016/j.mri.2013.07.004

    # calculate magnetization just before the WASABI pulse is applied (after T1prep pulse + trec delay)
    mzi = 1 - exp(-r1 * trec)  # magnetization

    m = mzi * ((da ** 2 * exp(-r1p * tp)) / dawi + (w1 ** 2 * cos(sqrt(dawi) * tp)) *
               exp(-r2p * tp) / dawi) + (r1 * da ** 2) / (r1p * dawi) * (1 - exp(-r1p * tp))

    return abs(m + (1 - m) * (1 - exp(-r1 * 0.00553)))  # relaxation during dead_time (30 µs) and spoiler (5.5 ms)


def wasabiti(x: Tensor,
             trec: Tensor,
             b0_shift: Tensor = 0.,
             b1: Tensor = 1.0,
             t1: Tensor = 2.,
             t2: Tensor = 0.1,
             freq: Tensor = 127.7292,
             tp: Tensor = 0.005,
             gamma: Tensor = 42.5764,
             ret_shape: Tensor = None
             ) -> Tensor:
    """Full Analytical WASABI approximation depending on b0_shift, B1, T1 and T2
    see: https://github.com/schuenke/WANTED/blob/main/fit/fit_functions.py

    :param x: frequency offsets [ppm]
    :param trec: recover time between different offsets/measurements. Can be a float or an array [s]
    :param b0_shift: b0_shift-shift [ppm]
    :param b1: relative B1 value []
    :param t1: longitudinal relaxation time (T_1) [s]
    :param t2: transversal relaxation time (T_2) [s]
    :param freq: frequency of the NMR system [MHz] (e.g. 128 for a 3T scanner)
    :param tp: duration of the WASABI pulse [s]
    :param gamma: gyromagnetic ratio [MHz/T]
    :param ret_shape: matrix filled with ones to be used as in-between to get the correct tensor multiplications
    """
    da = (x + ret_shape * b0_shift[:, None]) * freq * 2 * pi
    w1 = gamma * ret_shape * b1[:, None] * 2 * pi * 3.75
    dawi = da ** 2 + w1 ** 2

    r1 = 1. / (ret_shape * t1[:, None])
    r2 = 1. / (ret_shape * t2[:, None])
    r1p = (r1 * da ** 2) / (da ** 2 + w1 ** 2) + (r2 * w1 ** 2) / (da ** 2 + w1 ** 2)
    r2p = (2. * r2 + r1) / 2. - r1p / 2.  # according to https://doi.org/10.1016/j.mri.2013.07.004

    # calculate magnetization just before the WASABI pulse is applied (after T1prep pulse + trec delay)
    mzi = 1 - exp(-r1 * trec)  # magnetization

    return abs(mzi * ((da ** 2 * exp(-r1p * tp)) / dawi + (w1 ** 2 * cos(sqrt(dawi) * tp)) *
                      exp(-r2p * tp) / dawi) + (r1 * da ** 2) / (r1p * dawi) * (1 - exp(-r1p * tp)))


def zip_and_del_tb(filepath: str):
    """This function zips a tensorboard folder at the given path and deletes it afterwards.
    Intended to be used after training to enable a better git cycle.

    :param filepath: The filepath to the tensorboard folder.
    """
    if not isinstance(filepath, str):
        print("\033[0;30;41m The given filepath is not a string. \033[0m")
    elif os.path.isdir(os.path.join(filepath, 'tensorboard')):
        # create root path
        src = filepath
        filepath = os.path.join(filepath, 'tensorboard')

        # zip file
        with zipfile.ZipFile(filepath + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(filepath):
                for file in files:
                    # write while preserving relative structure to the tensorboard folder
                    zip_file.write(os.path.join(root, file), arcname=os.path.join(root, file)[len(src) + 1:])

        # remove tensorboard folder
        shutil.rmtree(filepath)
    else:
        print("\033[0;30;41m The given filepath does not exist. \033[0m")


def calc_print_b0shift(predictions: Tensor,
                       predictions_targets: Tensor,
                       index: int = 0) -> [float, float, float]:
    """Calculates values needed for the _print_b0shift() function. Can be used for separate evaluation but relies on
    the self.predictions and self.predictions_targets tensors.
    IMPORTANT If you don't have the parameters dB0, B1, T1 and T2 in that order you have to manually enter the
    correct index for the predictions array!

    :param predictions: like the class variable
    :param predictions_targets: like the class variable
    :param index: the index in the predictions array where B1 can be found
    :return: mean abs. dB0 error, max abs. dB0 error
    """
    mean_abs_db0 = mean(abs(predictions[:, index] -
                            predictions_targets[:, index])) * 1.
    max_abs_db0 = max(abs(predictions[:, index] -
                          predictions_targets[:, index])) * 1.
    mean_rel_db0 = mean(abs((predictions[:, index] -
                             predictions_targets[:, index]) /
                            predictions_targets[:, index])) * 100

    return mean_abs_db0, max_abs_db0, mean_rel_db0


def calc_print_b1(predictions: Tensor,
                  predictions_targets: Tensor,
                  index: int = 1, B1: float = 3.75) -> [float, float, float]:
    """Calculates values needed for the _print_b1() function. Can be used for separate evaluation but relies on the
    self.predictions and self.predictions_targets tensors. Get B1 value from the simulation. NOTE: here it
    is in µT, in simulation in T (-> factor 1E-6) \n
    IMPORTANT If you don't have the parameters dB0, B1, T1 and T2 in that order you have to manually enter the
    correct index for the predictions array!

    :param predictions: like the class variable
    :param predictions_targets: like the class variable
    :param index: the index in the predictions array where B1 can be found
    :return: mean abs. B1 error, max abs. B1 error, mean rel. B1 error
    """
    mean_abs_b1 = mean(abs(predictions[:, index] -
                           predictions_targets[:, index])) * B1
    max_abs_b1 = max(abs(predictions[:, index] -
                         predictions_targets[:, index])) * B1
    mean_rel_b1 = mean(abs((predictions[:, index] -
                            predictions_targets[:, index]) /
                           predictions_targets[:, index])) * 100

    return mean_abs_b1, max_abs_b1, mean_rel_b1


def calc_print_t1(predictions: Tensor,
                  predictions_targets: Tensor,
                  index: int = 2) -> [float, float, float]:
    """Calculates values needed for the _print_t1() function. Can be used for separate evaluation but relies on the
    self.predictions and self.predictions_targets tensors.
    IMPORTANT If you don't have the parameters dB0, B1, T1 and T2 in that order you have to manually enter the
    correct index for the predictions array!

    :param predictions: like the class variable
    :param predictions_targets: like the class variable
    :param index: the index in the predictions array where T1 can be found
    :return: mean abs. T1 error, max abs. T1 error, mean rel. T1 error
    """
    mean_abs_t1 = mean(abs(predictions[:, index] -
                           predictions_targets[:, index])) * 1000
    max_abs_t1 = max(abs(predictions[:, index] -
                         predictions_targets[:, index])) * 1000
    mean_rel_t1 = mean(abs((predictions[:, index] -
                            predictions_targets[:, index]) /
                           predictions_targets[:, index])) * 100

    return mean_abs_t1, max_abs_t1, mean_rel_t1


def calc_print_t2(predictions: Tensor,
                  predictions_targets: Tensor,
                  index: int = 3) -> [float, float, float]:
    """Calculates values needed for the _print_t2() function. Can be used for separate evaluation but relies on the
    self.predictions and self.predictions_targets tensors.
    IMPORTANT If you don't have the parameters dB0, B1, T1 and T2 in that order you have to manually enter the
    correct index for the predictions array!

    :param predictions: like the class variable
    :param predictions_targets: like the class variable
    :param index: the index in the predictions array where T2 can be found
    :return: mean abs. T2 error, max abs. T2 error, mean rel. T2 error
    """
    mean_abs_t2 = mean(abs(predictions[:, index] -
                           predictions_targets[:, index])) * 1000
    max_abs_t2 = max(abs(predictions[:, index] -
                         predictions_targets[:, index])) * 1000
    mean_rel_t2 = mean(abs((predictions[:, index] -
                            predictions_targets[:, index]) /
                           predictions_targets[:, index])) * 100

    return mean_abs_t2, max_abs_t2, mean_rel_t2


def get_min_loss_from_model(filepath: str) -> float:
    """Reads out the minimal validation loss of the model.
    :param filepath: the filepath to the """
    # load binary file into dictionary
    file_dict = load(filepath)
    # extract lowest val_loss
    return file_dict['overfit_check_tuple'][1]


def load_config_from_model(filepath: str) -> dict:
    """Loads the configuration file from the saved model.

    :param filepath: file path of the saved model.
    :return: dictionary with the full configuration file + its name as an item
    """
    # load binary file into dictionary
    file_dict = load(filepath)

    return file_dict['config']


def make_dir(dirpath: str, print_toggle=False):
    """Checks if the dirpath folder exists and if not creates it.

    :param dirpath: the path to check
    :param print_toggle: if True will print where a folder has been created
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        if print_toggle:
            print('created a new folder for the data: {}'.format(dirpath))


def load_config(filepath: str = 'net_config.yaml') -> dict:
    """Load config yaml file from path. It also creates the output path if it does not exist.

    :param filepath: path for the config file
    """
    with open(filepath) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    if not config.get('CONFIG_NAME'):
        if os.path.split(filepath)[-1].find('.') != -1:
            config['CONFIG_NAME'] = '.'.join(os.path.split(filepath)[-1].split('.')[:-1])
        else:
            print('WARNING: Your config name was not assigned correctly!')

    return config


def save_config(config: dict,
                directory: str = 'net_config',
                name: str = 'net_config.yaml'):
    """Save config yaml file to a path. It also creates the output path if it does not exist.

    :param config: config to be saved
    :param directory: path for the directory
    :param name: name of the config (with .yaml ending)
    """
    # check if given output path exists if not create it
    make_dir(directory, print_toggle=True)

    with open(os.path.join(directory, name),  'w') as f:
        yaml.dump(config, f)


