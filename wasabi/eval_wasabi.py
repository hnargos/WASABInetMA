# Standard library imports
import os
import time
from math import isnan
from typing import Optional, Union
from copy import deepcopy
import glob
import traceback
import csv

# Third party imports
import matplotlib.pyplot as plt
import torch

from torch import load, mean, abs, max, Tensor, zeros, no_grad, tensor, zeros_like
from torch import float32 as pytFl32
from numpy import ndarray, zeros as nzeros, abs as nabs, corrcoef as ncorrcoef
from sklearn.metrics import r2_score
from sklearn import linear_model

import tqdm

# our stuff
from .data import Data
from .neural_nets import DeepCEST_net, WASABI_net, WASABI_net2, DeepCEST_real_net, NeuralNet
from .loss import GNLL
from .auxiliary_functions import make_dir, load_config, load_config_from_model
from .auxiliary_functions import calc_print_b0shift as aux_calc_print_b0shift, calc_print_b1 as aux_calc_print_b1, \
    calc_print_t1 as aux_calc_print_t1, calc_print_t2 as aux_calc_print_t2
from .trainer import Trainer


def check_for_type(xy: Union[str, Tensor, ndarray]) -> Tensor:
    """Checks if it is one of these data types and converts to Tensor if necessary.
    :param xy: contains the path to the data or the data to be used for evaluation.
    :return: A tensor of the data.
    """
    if isinstance(xy, str):
        xy = load(xy)
        if isinstance(xy, ndarray):  # make tensor
            xy = tensor(xy, dtype=pytFl32)
        elif isinstance(xy, Tensor):
            pass
        else:
            raise TypeError('Your data in trainer does not have the correct type, '
                            'please use only torch.tensor or numpy.ndarray.')
        return xy
    elif isinstance(xy, ndarray):
        return tensor(xy, dtype=pytFl32)  # make tensor
    elif isinstance(xy, Tensor):
        return xy.type(dtype=pytFl32)
    else:
        raise TypeError('Your data has the wrong type. Please enter either a '
                        'str, pytorch Tensor or a numpy.ndarray.')


class EvalWasabi:
    """Contains the tools to evaluate the results of Trainer.
    """
    def __init__(self,
                 trainer: Union[str, dict, Trainer],
                 data_type: str,
                 net_to_load: Union[str, list] = None,
                 x_file: Union[str, Tensor, ndarray] = None,
                 y_file: Union[str, Tensor, ndarray] = None,
                 create_zeros_tgts: bool = False,
                 phantom_filepath: str = None,
                 offsets: int = 31,
                 use_tb: bool = False,
                 evaluation: bool = True,
                 add_noise: bool = False,
                 noise_scale: float = 1.):
        """

        :param trainer: str: path to a .pt neural network. dict: the config file. Trainer: a Trainer object where everything is loaded as wished. If it loads, it loads the "best_model" save from the configs output directory.
        :param data_type: 'BMCTool' for tensors of the form [n_samples, m_offsets], 'phantom': for data from BMCPhantom, 'scanner': for data from a scanner, [x_pixel, y_pixel, offsets].
        :param net_to_load: If list: it assumes the analysis is to be done with a ensemble for this the trainer parameter has to be a str or dict. If str: If you don't want to load the first 'best_model_save.pt' in your configs output directory specify here. IMPORTANT: does not do anything if a Trainer Object is handed over in 'trainer'.
        :param x_file: str: overwrites the DATA_X_PATH filepath in the config. Tensor: uses this data instead of the DATA_X_PATH filepath in the config.
        :param y_file: str: overwrites the DATA_Y_PATH filepath in the config. Tensor: uses this data instead of the DATA_Y_PATH filepath in the config.
        :param create_zeros_tgts: if this is set to True, it will ignore y_file and DATA_Y_PATH and assume no targets exist. It then creates a zero tensor for the targets.
        :param phantom_filepath: has to be filled if data_type is 'phantom'
        :param offsets: the offsets measured in the z-spectrum
        :param evaluation: only touch this parameter if you know what you are doing, it changes the evaluation parameters in data and trainer.
        :param use_tb: using tensorboard a few more statistics will be captured, requires tensorboard to be installed
        :param add_noise: Default: False; If True adds the normal noise. (ONLY for the phantom)
        :param noise_scale: Gets multiplied with the noise before it gets added to the x input.
        """
        self.data = Data()
        self.data_type = data_type
        self.data_shape = None  # relevant for the scanner data
        self.test_loss = []  # loss of the test set

        # transform to tensor
        if x_file is not None:
            x_file = check_for_type(x_file)
            if data_type == 'scanner':
                # save shape
                self.data_shape = [x_file.shape[0], x_file.shape[1]]
                # scanner data has the wrong shape
                x_file = x_file.reshape(-1, offsets)

        # transform to tensor
        if y_file is not None:
            y_file = check_for_type(y_file)

        # check type
        if not isinstance(create_zeros_tgts, bool):
            raise TypeError('existing_tgts has to be either True or False.')

        if isinstance(trainer, str) or isinstance(trainer, dict):
            # check type of data
            if data_type == 'BMCTool':
                self.data.load_data_tensor(config=trainer,
                                           x_file=x_file,
                                           y_file=y_file,
                                           printing=False,
                                           evaluation=evaluation,
                                           add_noise=True,
                                           create_zeros_tgts=create_zeros_tgts,
                                           noise_scale=noise_scale)
            elif data_type == 'scanner':
                self.data.load_data_tensor(config=trainer,
                                           x_file=x_file,
                                           y_file=y_file,
                                           printing=False,
                                           evaluation=evaluation,
                                           add_noise=False,
                                           create_zeros_tgts=create_zeros_tgts)

            elif data_type == 'phantom':
                self.data.load_phantom(filepath=phantom_filepath,
                                       config=trainer,
                                       add_noise=add_noise,
                                       noise_scale=noise_scale,)
            else:
                raise TypeError('The data_type variable has to be either BMCTool, phantom or scanner.')

            # load saved neural net
            if net_to_load is None:
                net_to_load = os.path.join(self.data.config['OUTPUT_DIR'], self.data.config['CONFIG_NAME'])
                net_to_load = glob.glob(net_to_load + '_best_model_save.pt')
                # ensure only one net is loaded if not warn user
                if len(net_to_load) > 1:
                    print('WARNING, there are multiple best_model_saves.pt in your directory. It loaded: {}. '
                          'If you want to load another one you will have to do that manually and hand '
                          'EvalWasabi a trainer object.'.format(net_to_load[0]))
                if len(net_to_load) == 0:
                    if os.path.exists(os.path.join(self.data.config['OUTPUT_DIR'],
                                                   self.data.config['CONFIG_NAME'],
                                                   self.data.config['CONFIG_NAME'],
                                                   '_best_model_save.pt')):
                        net_to_load = [os.path.join(self.data.config['OUTPUT_DIR'],
                                                    self.data.config['CONFIG_NAME'],
                                                    self.data.config['CONFIG_NAME'],
                                                    '_best_model_save.pt')]
                    else:
                        raise ValueError('Did not find a saved net. In older configs the issue is that the output '
                                         'paths are structured differently and this, just add the path manually to '
                                         'net_to_load=')
                net_to_load = net_to_load[0]

            # prep and load ensemble
            if isinstance(net_to_load, list):
                tmp_trainer_list = []
                for i in range(len(net_to_load)):
                    # initialize Trainer
                    tmp_trainer = Trainer(data=self.data, use_tb=use_tb)
                    tmp_trainer.load_net(net_to_load[i])
                    tmp_trainer_list.append(tmp_trainer)

                self.trainer = tmp_trainer_list
            # assert incorrect type
            elif not(isinstance(net_to_load, str) or isinstance(net_to_load, list) or net_to_load is None):
                raise TypeError('net_to_load has to be a str, a list or None. It is {}'.format(type(net_to_load)))
            # load single net
            else:
                # initialize Trainer
                self.trainer = [Trainer(data=self.data, use_tb=use_tb)]
                self.trainer[0].load_net(net_to_load)

        # trainer already is a Trainer() object, fill class variables
        elif isinstance(trainer, Trainer):
            self.data = trainer.data
            self.trainer = [trainer]

            # refill data and trainer is different data is given
            if x_file is not None or y_file is not None:
                # check type of data
                if data_type == 'BMCTool':
                    self.data.load_data_tensor(config=self.data.config,
                                               x_file=x_file,
                                               y_file=y_file,
                                               printing=False,
                                               evaluation=evaluation,
                                               add_noise=True,
                                               create_zeros_tgts=create_zeros_tgts)

                    # refill trainer
                    self.trainer[0].train_loader = self.data.train_loader
                    self.trainer[0].val_loader = self.data.val_loader
                    self.trainer[0].test_loader = self.data.test_loader

                elif data_type == 'scanner':
                    self.data.load_data_tensor(config=self.data.config,
                                               x_file=x_file,
                                               y_file=y_file,
                                               printing=False,
                                               evaluation=evaluation,
                                               add_noise=False,
                                               create_zeros_tgts=create_zeros_tgts)

                    # refill trainer
                    self.trainer[0].train_loader = self.data.train_loader
                    self.trainer[0].val_loader = self.data.val_loader
                    self.trainer[0].test_loader = self.data.test_loader

                elif data_type == 'phantom':
                    self.data.load_phantom(filepath=phantom_filepath,
                                           config=self.data.config)
                else:
                    raise TypeError('The data_type variable has to be either BMCTool, phantom or scanner.')
        else:
            raise TypeError('trainer has the wrong data type, it has to be either a str, a dict or a Trainer object.')

        # keep net names for ensemble
        self.net_to_load = net_to_load

        assert offsets == self.data.config['N_NEURONS'][0], "Your given offsets differ from the the offsets in the net."

    def _predict(self,
                 trainer: Trainer,
                 show_test_set_loss: bool = False,
                 ret: bool = False) -> float:
        """Evaluates the NN using the test data.
        It can prints the results, and copies the predictions to self.predictions and
        the targets to self.prediction_targets.

        :param trainer: Trainer object to be predicted
        :param show_test_set_loss: WARNING this should never be used to choose which trained network is used. For that the validation values exist. This is purely for evaluating the dataset on entirely new data, to give a accurate estimate of the quality of the results.
        :param ret: if true then this function returns the loss of the test set
        :return: mean loss of the test set
        """
        # create test_loader progress bar
        tqdm_test_loader = tqdm.tqdm(trainer.test_loader, ascii=True, position=0)

        # turn gradients off
        with no_grad():
            # eval and write predictions to class variable
            n_samples, cuml_loss_epoch, _ = trainer._core_loop(data_loader=tqdm_test_loader,
                                                               loader_type='test')

        if show_test_set_loss:
            print('test set loss:', cuml_loss_epoch / n_samples, '\n')

        if ret:
            return cuml_loss_epoch / n_samples

    def predict(self,
                show_test_set_loss: bool = False,
                ret: bool = False) -> Union[float, list]:
        """Evaluates the NN using the test data.
        It can prints the results, and copies the predictions to self.predictions and
        the targets to self.prediction_targets.

        :param show_test_set_loss: WARNING this should never be used to choose which trained network is used. For that the validation values exist. This is purely for evaluating the dataset on entirely new data, to give a accurate estimate of the quality of the results.
        :param ret: if true then this function returns the loss of the test set
        :return: mean loss of the test set(s)
        """
        if isinstance(self.trainer, list):
            # ensemble
            ret_val = []
            for i in range(len(self.trainer)):
                ret_val.append(self._predict(self.trainer[i], show_test_set_loss, True))

            # copy to class variable to use else where
            self.test_loss = ret_val.copy()
            if ret:
                return ret_val
        else:
            raise ValueError('Could net recognize self.trainer object.')

    def get_predictions(self, ens: bool = False, uncer_type: str = "aleatoric") -> Tensor:
        """
        Returns the predictions from the trainer objects in a tensor form. It may be necessary to call Tensor.cpu()
        for other calculations or before converting with Tensor.numpy().

        :param ens: If False: The tensor contains the mean of all predictions and the mean of all the variances for that value. If True: The Tensor contains the mean of the prediction and the variance of the predictions for that value.
        :param uncer_type: Default: "aleatoric", always used on pure GNLL, uncertainty is the mean of all uncertainties in the ensemble; "epistemic, uncertainty is the standard deviation of all uncertainties; "combined", adds up both uncertainties
        :return: Tensor(N, param : uncertainty == standart deviation)
        """
        if not ens:
            # calculate the mean for mean an variance of all trainer objects
            ret_val = zeros_like(self.trainer[0].predictions, device=self.data.dev)
            for i in range(len(self.trainer)):
                ret_val += self.trainer[i].predictions
            return ret_val.detach()/len(self.trainer)
        else:
            print("evaluating ensemble mean and variance")
            # calculate the ensemble mean and variance of all trainer objects
            ret_val_tmp = zeros(list(self.trainer[0].predictions.shape) + [len(self.trainer)], device=self.data.dev)
            ret_val = zeros_like(self.trainer[0].predictions, device=self.data.dev)
            for i in range(len(self.trainer)):
                ret_val_tmp[:, :, i] = self.trainer[i].predictions

            # mean of param
            ret_val[:, :self.data.n_tgt_params] = ret_val_tmp[:, :self.data.n_tgt_params, :].mean(dim=2)

            # TODO fix uncertainty to first do single nets and then mean(var) and std
            if uncer_type == "aleatoric":
                # std = sqrt(mean(sqr(std))) (aleatoric uncertainty)
                ret_val[:, self.data.n_tgt_params:] = (ret_val_tmp[:, self.data.n_tgt_params:, :].square().mean(dim=2)).sqrt()
                # OLD
                # ret_val[:, self.data.n_tgt_params:] = ret_val_tmp[:, :self.data.n_tgt_params, :].std(dim=2)
            elif uncer_type == "epistemic":
                # std of the ensemble mean
                ret_val[:, self.data.n_tgt_params:] = (ret_val_tmp[:, :self.data.n_tgt_params, :].square().mean(dim=2) -
                                                       ret_val_tmp[:, :self.data.n_tgt_params, :].mean(dim=2).square()).abs().sqrt()
            elif uncer_type == "combined":
                # combined of the two options above
                ret_val[:, self.data.n_tgt_params:] = (ret_val_tmp[:, self.data.n_tgt_params:, :].square().mean(dim=2) +
                                                       ret_val_tmp[:, :self.data.n_tgt_params, :].square().mean(dim=2) -
                                                       ret_val_tmp[:, :self.data.n_tgt_params, :].mean(dim=2).square()).sqrt()
            else:
                raise ValueError("The get_predictions function has received {}, acceptable are: aleatoric, epistemic, combined.".format(uncer_type))
            return ret_val.detach()

    def calc_print_b0shift(self, trainer: list = None,  index: int = 0) -> (float, float, float):
        """Calculates an evaluation of dB0."""
        if trainer is None:
            trainer = self.trainer

        if isinstance(trainer, list):
            # calculate the mean of all predictions in the ensemble
            pred = zeros_like(trainer[0].predictions)
            for i in range(len(trainer)):
                pred += trainer[i].predictions
            pred /= len(trainer)

            # calculate the parameter check
            return aux_calc_print_b0shift(pred, trainer[0].predictions_targets, index)
        else:
            raise ValueError("trainer has to be list type.")

    def _print_b0shift(self, mean_abs_db0, max_abs_db0, mean_rel_db0):
        """Prints an evaluation of the b0shift."""

        print('INFO: dB0 error should not exceed 0.05 ppm')
        print(f'''mean abs. dB0 error (goal <0.008) = {mean_abs_db0:.4f} ppm''')
        print(f'''max abs. dB0 error (goal <2.881) = {max_abs_db0:.4f} ppm''')
        print(f'''mean rel. dB0 error = {mean_rel_db0:.4f} %''')
        print(' ')

    def calc_print_b1(self, trainer: list = None,  index: int = 1) -> (float, float, float):
        """Calculates an evaluation of B1."""
        if trainer is None:
            trainer = self.trainer

        if isinstance(trainer, list):
            # calculate the mean of all predictions in the ensemble
            pred = zeros_like(trainer[0].predictions)
            for i in range(len(trainer)):
                pred += trainer[i].predictions
            pred /= len(trainer)

            # calculate the parameter check
            return aux_calc_print_b1(pred, trainer[0].predictions_targets, index)
        else:
            raise ValueError("trainer has to be list type.")

    def _print_b1(self, mean_abs_b1, max_abs_b1, mean_rel_b1):
        """Prints an evaluation of the B1."""

        print('INFO: abs B1 error should not exceed 0.1 µT')
        print(f'''mean abs. B1 error (goal <0.124) = {mean_abs_b1:.4f} µT''')
        print(f'''max abs. B1 error (goal <4.774) = {max_abs_b1:.4f} µT''')
        print(f'''mean rel. B1 error = {mean_rel_b1:.4f} %''')
        print(' ')

    def calc_print_t1(self, trainer: list = None,  index: int = 2) -> (float, float, float):
        """Calculates an evaluation of T1."""
        if trainer is None:
            trainer = self.trainer

        if isinstance(trainer, list):
            # calculate the mean of all predictions in the ensemble
            pred = zeros_like(trainer[0].predictions)
            for i in range(len(trainer)):
                pred += trainer[i].predictions
            pred /= len(trainer)

            # calculate the parameter check
            return aux_calc_print_t1(pred, trainer[0].predictions_targets, index)
        else:
            raise ValueError("trainer has to be list type.")

    def _print_t1(self, mean_abs_t1, max_abs_t1, mean_rel_t1):
        """Prints an evaluation of the T1."""

        print('INFO: abs T1 error should not exceed 50 ms')
        print(f'''mean abs. T1 error = {mean_abs_t1:.1f} ms''')
        print(f'''max abs. T1 error = {max_abs_t1:.1f} ms''')
        print(f'''mean rel. T1 error = {mean_rel_t1:.3f} %''')
        print(' ')

    def calc_print_t2(self, trainer: list = None,  index: int = 3) -> (float, float, float):
        """Calculates an evaluation of T2."""
        if trainer is None:
            trainer = self.trainer

        if isinstance(trainer, list):
            # calculate the mean of all predictions in the ensemble
            pred = zeros_like(trainer[0].predictions)
            for i in range(len(trainer)):
                pred += trainer[i].predictions
            pred /= len(trainer)

            # calculate the parameter check
            return aux_calc_print_t2(pred, trainer[0].predictions_targets, index)
        else:
            raise ValueError("trainer has to be list type.")

    def _print_t2(self, mean_abs_t2, max_abs_t2, mean_rel_t2):
        """Prints an evaluation of the T2."""

        print('INFO: abs T2 error should not exceed ? ms')
        print(f'''mean abs. T2 error = {mean_abs_t2:.1f} ms''')
        print(f'''max abs. T2 error = {max_abs_t2:.1f} ms''')
        print(f'''mean rel. T2 error = {mean_rel_t2:.3f} %''')
        print(' ')

    def r2(self, param: str, ens: bool = False, uncer_type: str = "aleatoric", **kwargs) -> float:
        """Returns the R² (coefficient of determination) value of the predictions using sklearn.metrics.r2_score.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

        :param param: dB0 or b0_shift, B1 or b1_inhom, T1 or t1, T2 or t2
        :param kwargs: kwargs used by sklearn.metrics.r2_score
        :param ens:  If False: The tensor contains the mean of all predictions and the mean of all the variances for that value. If True: The Tensor contains the mean of the prediction and the variance of the variances for that value.
        :param uncer_type: Default: "aleatoric", always used on pure GNLL, uncertainty is the mean of all uncertainties in the ensemble; "epistemic, uncertainty is the standard deviation of all uncertainties; "combined", adds up both uncertainties
        :return: the R² value
        """
        # set param
        net_labels = self.trainer[0].config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2'])
        if param not in net_labels and param in ['b0_shift', 'b1_inhom', 't1', 't2']:
            param = {'b0_shift': 'dB0', 'b1_inhom': 'B1', 't1': 'T1', 't2': 'T2'}[param]
        elif param in net_labels:
            pass
        else:
            raise ValueError('param has a value that cannot be interpreted: {}. It is possible the network'
                             ' is not trained for that parameter'.format(param))

        # select
        for i, j in enumerate(net_labels):
            if param == j:
                # calculate the mean of all predictions in the ensemble
                pred = self.get_predictions(ens=ens, uncer_type=uncer_type)

                return r2_score(y_true=self.trainer[0].predictions_targets[:, i].detach().cpu().numpy().copy(),
                                y_pred=pred[:, i].detach().cpu().numpy().copy(),
                                **kwargs)

        raise RuntimeError('An unexpected error has occurred. EvalWasabi.r2 could not select right parameter.')

    def pearson(self, param: str) -> float:
        """
        :param param: dB0, B1, T1 or T2
        :return: the pearson correlation between the predicted and true value
        """
        index = None
        # check for ensemble
        if len(self.trainer) > 1:
            ens = True
        else:
            ens = False

        # get predictions for value
        for i, j in enumerate(self.trainer[0].config.get('TYPE_PARAMS')):
            if param == j:
                index = i
                pred = self.get_predictions(ens=ens)[:, i]

        # ensure a lacking index is noticed
        if index is None:
            return 99.

        # assign targets
        pred_tgts = self.trainer[0].predictions_targets[:, index]

        # calc pearson correlation
        pears = ncorrcoef(pred_tgts.detach().cpu().numpy().copy(),
                          pred.detach().cpu().numpy().copy())[0, 1]

        return pears

    def _save_to_csv(self, trainer: list, net_name: str = None):
        """Saves param_min_max_check values to output/param_check.csv. Removes previous entries with the same name.

        :param net_name: if specified will use this as the name identifier when saving
        """
        # check for ensemble
        if len(self.trainer) > 1:
            ens = True
        else:
            ens = False

        # TODO modify with coefficient of determination + MSE + std for each parameter and the dataset as a whole
        # TODO a measure of quality for the uncertainty vs difference
        ## prepare save
        # set header for table and create file
        if not os.path.exists('output/param_check.csv'):
            with open('output/param_check.csv', 'w') as f:
                write = csv.writer(f)
                write.writerow(['name',
                                'mean_abs_dB0', 'max_abs_dB0', 'mean_rel_dB0', 'dB0_r2', 'dB0_pears', 'dB0_std',
                                'mean_abs_B1', 'max_abs_B1', 'mean_rel_B1', 'B1_r2', 'B1_pears', 'B1_std',
                                'mean_abs_T1', 'max_abs_T1', 'mean_rel_T1', 'T1_r2', 'T1_pears', 'T1_std',
                                'mean_abs_T2', 'max_abs_T2', 'mean_rel_T2', 'T2_r2', 'T2_pears', 'T2_std',
                                'loss mean', 'loss std'])
        body = []

        if net_name is None:
            net_name = str(self.data.config.get('CONFIG_NAME'))

        body += [net_name]

        # prep body
        for i, j in enumerate(['dB0', 'B1', 'T1', 'T2']):
            if j == 'dB0':
                if 'dB0' in trainer[0].config.get('TYPE_PARAMS'):
                    for k, l in enumerate(trainer[0].config.get('TYPE_PARAMS')):
                        if l == 'dB0':
                            body += list(self.calc_print_b0shift(index=k))
                            body += [self.r2('dB0')]
                            body += [self.pearson('dB0')]
                            body += [(self.get_predictions(ens=ens)[:, k] -
                                      self.trainer[0].predictions_targets[:, k]).abs().std().item() * 1.]
                else:
                    body += ['0.', '0.', '0.', '0.', '0', '0']
            elif j == 'B1':
                if 'B1' in trainer[0].config.get('TYPE_PARAMS'):
                    for k, l in enumerate(trainer[0].config.get('TYPE_PARAMS')):
                        if l == 'B1':
                            body += list(self.calc_print_b1(index=k))
                            body += [self.r2('B1')]
                            body += [self.pearson('B1')]
                            body += [(self.get_predictions(ens=ens)[:, k] -
                                      self.trainer[0].predictions_targets[:, k]).abs().std().item() * 3.75]
                else:
                    body += ['0.', '0.', '0.', '0.', '0', '0']
            elif j == 'T1':
                if 'T1' in trainer[0].config.get('TYPE_PARAMS'):
                    for k, l in enumerate(trainer[0].config.get('TYPE_PARAMS')):
                        if l == 'T1':
                            body += list(self.calc_print_t1(index=k))
                            body += [self.r2('T1')]
                            body += [self.pearson('T1')]
                            body += [(self.get_predictions(ens=ens)[:, k] -
                                      self.trainer[0].predictions_targets[:, k]).abs().std().item() * 1000.]
                else:
                    body += ['0.', '0.', '0.', '0.', '0', '0']
            elif j == 'T2':
                if 'T2' in trainer[0].config.get('TYPE_PARAMS'):
                    for k, l in enumerate(trainer[0].config.get('TYPE_PARAMS')):
                        if l == 'T2':
                            body += list(self.calc_print_t2(index=k))
                            body += [self.r2('T2')]
                            body += [self.pearson('T2')]
                            body += [(self.get_predictions(ens=ens)[:, k] -
                                      self.trainer[0].predictions_targets[:, k]).abs().std().item() * 1000.]
                else:
                    body += ['0.', '0.', '0.', '0.', '0', '0']
            else:
                raise NameError('{} is not dB0, B1, T1 or T2. This function only exists for these values'.format(j))

        # add loss
        body += [tensor(self.test_loss).mean().item(), tensor(self.test_loss).std().item()]

        # transform all pytorch tensors to floats
        for i in range(1, len(body)):
            body[i] = float(body[i])

        # read existing file
        with open('output/param_check.csv', 'r') as f:
            csv_reader = csv.reader(f)
            csv_content = []

            for i in csv_reader:
                csv_content += [i]

        # delete possible duplicate
        index = 0
        for i in range(len(csv_content)):
            if csv_content[i][0] == net_name:
                index = i
        if index != 0:
            del csv_content[index]

        # overwrite old file with new line appended
        csv_content += [body]
        with open('output/param_check.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(csv_content)

    def _parameter_min_max_check(self,
                                 save_file: bool = False,
                                 print_param: bool = True,
                                 net_name: str = None,
                                 loss: bool = False):
        """Calculates the mean and max values of the three parameters dB0, B1, T1 and T2 and prints them in context. \n
        If this is to evaluate a ensemble, all the various predictions get averaged before the check is calculated.

        :param save_file: if True will save the params into output/param_check.csv
        :param print_param: if False will not print anything to console
        :param net_name: if specified will use this as the name identifier when saving
        :param loss: if the loss is saved as well, in case of an ensemble it calculates the mean & std
        """
        # save eval locally
        body = []

        # check if the config specifies the values otherwise assume the standard order of dB0, B1, T1, T2
        for i, j in enumerate(self.trainer[0].config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2'])):
            if j == 'dB0':
                v1, v2, v3 = self.calc_print_b0shift(index=i)
                if print_param:
                    self._print_b0shift(v1, v2, v3)
            elif j == 'B1':
                v1, v2, v3 = self.calc_print_b1(index=i)
                if print_param:
                    self._print_b1(v1, v2, v3)
            elif j == 'T1':
                v1, v2, v3 = self.calc_print_t1(index=i)
                if print_param:
                    self._print_t1(v1, v2, v3)
            elif j == 'T2':
                v1, v2, v3 = self.calc_print_t2(index=i)
                if print_param:
                    self._print_t2(v1, v2, v3)
            else:
                raise NameError('{} is not dB0, B1, T1 or T2. This function only exists for these values'.format(j))

            body += ['mean_abs_{}: {:.3e}, max_abs_{}: {:.3e}, mean_rel_{}: {:.3e}\n'.format(j, v1, j, v2, j, v3)]

        # add loss in param_check
        if loss:
            body += ['test loss mean: {:.3e}\n test loss std: {:.3e}'.format(tensor(self.test_loss).mean(),
                                                                             tensor(self.test_loss).std())]
            if print_param:
                print(['test loss mean: {:.3e}\n test loss std: {:.3e}'.format(tensor(self.test_loss).mean(),
                                                                               tensor(self.test_loss).std())])

        # choose filepath for local save
        if net_name is not None:
            filepath_local = net_name
        elif self.net_to_load is not None and isinstance(self.net_to_load, list):
            filepath_local = self.net_to_load[0]
        else:
            filepath_local = self.net_to_load

        filepath_local = os.path.join(os.path.split(filepath_local)[0], 'param_check.txt')
        # save locally
        with open(filepath_local, 'w') as f:
            f.write(''.join(body))

        # save
        if save_file:
            self._save_to_csv(self.trainer, net_name=net_name)

    def parameter_min_max_check(self, save_file: bool = False,
                                print_param: bool = True,
                                net_name: str = None,
                                loss: bool = False):
        """Calculates the mean and max values of the three parameters dB0, B1, T1 and T2 and prints them in context. \n
        If

        :param save_file: if True will save the params into output/param_check.csv
        :param print_param: if False will not print anything to console
        :param net_name: if specified will use this as the name identifier when saving
        :param loss: if the loss is saved as well, in case of an ensemble it calculates the mean & std
        """
        self._parameter_min_max_check(save_file, print_param, net_name, loss)

    def _make_matrix_for_phantom(self, index: int,
                                 uncert: bool = False,
                                 ens: bool = False,
                                 uncer_type: str = "aleatoric",
                                 tgt: bool = False) -> Tensor or [Tensor, Tensor]:
        """Takes the self.predictions variable and outputs a square tensor with each prediction in the correct place
        for an image.

        :param index: of the self.predictions variable to be used.
        Order: dB0, B1, T1, T2, dB0 std, B1 std, T1 std, T2 std
        :param uncert: if true returns [Tensor, Tensor] with the second one being the uncertainties
        :param ens:  If False: The tensor contains the mean of all predictions and the mean of all the variances for that value. If True: The Tensor contains the mean of the prediction and the variance of the variances for that value.
        :param uncer_type: Default: "aleatoric", always used on pure GNLL, uncertainty is the mean of all uncertainties in the ensemble; "epistemic, uncertainty is the standard deviation of all uncertainties; "combined", adds up both uncertainties
        :param tgt: Default: False; If true returns the true targets as matrix.
        :return: a square pytorch tensor with the predictions, if uncert is true returns [Tensor, Tensor] with the second one beeing the uncertainties
        """
        matrix_size = len(self.data.raw_data['phantom']['t1'])  # get matrix size

        # create and fill the z spectra matrix with loaded values or zero spectra
        z_specs_tensor = zeros((matrix_size, matrix_size), dtype=pytFl32, device=self.data.dev)
        z_specs_tensor_uncert = zeros((matrix_size, matrix_size), dtype=pytFl32, device=self.data.dev)
        if tgt:
            predictions = self.trainer[0].predictions_targets
        else:
            predictions = self.get_predictions(ens=ens, uncer_type=uncer_type)

        k = 0  # counter for the flat predictions array
        for i in range(matrix_size):
            for j in range(matrix_size):
                # add prediction based on original z_spectra matrix from the phantom data
                if self.data.raw_data['z_specs'].get((i, j)) is not None:
                    z_specs_tensor[i, j] += predictions[k, index].detach()
                    if uncert:
                        z_specs_tensor_uncert[i, j] += predictions[k, index + self.data.n_tgt_params].detach()
                    k += 1
        # average over ensemble and return
        if uncert:
            return [z_specs_tensor.cpu(), z_specs_tensor_uncert.cpu()]
        return z_specs_tensor.cpu()

    def get_map(self, param: str,
                predict: bool = False,
                uncert: bool = False,
                ens: bool = False,
                uncer_type: str = "aleatoric",
                tgt: bool = False) -> ndarray:
        """Returns a parameter map for the phantom.

        :param param: "dB0", "B1", "T1", "T2", "b0_shift" will be considered as "dB0" and "b1_inhom" as "B1"
        :param predict: If true will first predict the dataset and then create the map.
        :param uncert: if true returns a map of the parameters uncertainty
        :param ens:  If False: The tensor contains the mean of all predictions and the mean of all the variances for that value. If True: The Tensor contains the mean of the prediction and the variance of the variances for that value.
        :param uncer_type: Default: "aleatoric", always used on pure GNLL, uncertainty is the mean of all uncertainties in the ensemble; "epistemic, uncertainty is the standard deviation of all uncertainties; "combined", adds up both uncertainties
        :param tgt: Default: False; If true returns the true targets as a matrix. Does not work if uncert=True. Needs data_type='phantom'.
        :return: the map as a matrix
        """
        # predict the dataset
        if predict:
            self.predict()

        net_labels = ['dB0', 'B1', 'T1', 'T2']
        if param not in net_labels and param in ['b0_shift', 'b1_inhom', 't1', 't2']:
            param = {'b0_shift': 'dB0', 'b1_inhom': 'B1', 't1': 'T1', 't2': 'T1'}[param]
        elif param not in net_labels:
            raise ValueError('param {} is not a valid map parameter.'.format(param))

        # use config or BMCTool default order
        type_params = self.trainer[0].config.get('TYPE_PARAMS', net_labels)

        if self.data_type == 'phantom':
            # create param map
            for i, j in enumerate(type_params):
                if param == j:
                    if uncert:
                        return self._make_matrix_for_phantom(i, uncert=True, ens=ens,
                                                             uncer_type=uncer_type)[1].detach().numpy().copy()
                    if tgt:
                        return self._make_matrix_for_phantom(i, ens=ens, uncer_type=uncer_type, tgt=True).detach().numpy().copy()
                    return self._make_matrix_for_phantom(i, ens=ens, uncer_type=uncer_type).detach().numpy().copy()

        elif self.data_type == 'scanner':
            # create param map
            ret = nzeros(self.data_shape + [int(self.data.n_tgt_params * 2)])
            for i, j in enumerate(type_params):
                if param == j:
                    if uncert:
                        # consider possible ensemble
                        ret = self.get_predictions(ens=ens, uncer_type=uncer_type).reshape(self.data_shape +
                                                                    [int(self.data.n_tgt_params * 2)]
                                                                    )[:, :, i + self.data.n_tgt_params
                                                                      ].detach().cpu().numpy().copy()
                    elif tgt:
                        # returns true targets
                        # does not return full ensemble length, only single data length
                        ret = self.trainer[0].predictions_targets.detach().cpu().numpy().copy()
                    else:
                        ret = self.get_predictions(ens=ens, uncer_type=uncer_type).reshape(self.data_shape +
                                                                    [int(self.data.n_tgt_params * 2)]
                                                                    )[:, :, i].detach().cpu().numpy().copy()
                    return ret

        else:
            raise ValueError("The map could not be produced. Your data_type has to be phantom or scanner.")

    def get_all_maps(self, predict: bool = False,
                     param_labels: str = 'BMCPhantom',
                     uncert: bool = False,
                     ens: bool = False,
                     uncer_type: str = "aleatoric") -> [dict, dict]:
        """Returns the maps in order as specified in the config for the NN.
        :param predict: If true will first predict the dataset and then create the map.
        :param param_labels: "BMCPhantom" for [b0_shift, b1_inhom, t1, t2] or "WASABInet" for [dB0, B1, T1, T2]
        :param uncert: if true returns two dictionaries, the second being the uncertainties with the same labels as the first
        :param ens:  If False: The tensor contains the mean of all predictions and the mean of all the variances for that value. If True: The Tensor contains the mean of the prediction and the variance of the variances for that value.
        :param uncer_type: Default: "aleatoric", always used on pure GNLL, uncertainty is the mean of all uncertainties in the ensemble; "epistemic, uncertainty is the standard deviation of all uncertainties; "combined", adds up both uncertainties
        :return: returns the maps as a dict with keys as defined by param_labels
        """
        ret = {}  # return dict
        ret_uncert = {}  # contains uncertainties if required
        labels = {'WASABInet': ['dB0', 'B1', 'T1', 'T2'],
                  'BMCPhantom': ['b0_shift', 'b1_inhom', 't1', 't2']}
        assert param_labels == 'WASABInet' or param_labels == 'BMCPhantom', "param_labels has to be WASABInet or " \
                                                                            "BMCPhantom "

        # predict the dataset
        if predict:
            self.predict()

        # iterate over all parameter to get all maps
        for param in self.trainer[0].config.get('TYPE_PARAMS', labels['WASABInet']):
            ret.update({labels[param_labels][labels['WASABInet'].index(param)]: self.get_map(param, ens=ens)})
            if uncert:
                ret_uncert.update({labels[param_labels][labels['WASABInet'].index(param)]: self.get_map(param,
                                                                                                        uncert=True,
                                                                                                        ens=ens,
                                                                                                        uncer_type=uncer_type)})

        if uncert:
            return ret, ret_uncert
        return ret

    def _plot_parameter(self, param: str,
                        title: str = '',
                        xlabel: str = '',
                        ylabel: str = '',
                        predict: bool = False,
                        uncert: bool = False,
                        ens: bool = False,
                        uncer_type: str = "aleatoric",
                        tgt: bool = False,
                        diff: bool = False,
                       save_fig: str = False):
        """Creates a plot of the requested parameter or uncertainty.

        :param param: "dB0", "B1", "T1", "T2", "b0_shift" will be considered as "dB0" and "b1_inhom" as "B1"
        :param title: title of the plot
        :param xlabel: label of the x axis
        :param ylabel: label of the y axis
        :param predict: If true will first predict the dataset and then create the map.
        :param uncert: if true returns a map of the parameters uncertainty
        :param ens:  If False: The tensor contains the mean of all predictions and the mean of all the variances for that value. If True: The Tensor contains the mean of the prediction and the variance of the variances for that value.
        :param uncer_type: Default: "aleatoric", always used on pure GNLL, uncertainty is the mean of all uncertainties in the ensemble; "epistemic, uncertainty is the standard deviation of all uncertainties; "combined", adds up both uncertainties
        :param tgt: Default: False; If true returns the targets as a matrix. Does not work if uncert=True. Needs self.data_type='phantom'.
        :param diff: Default: False; If true returns a plot of the difference between target and NN result.
        :param save_fig: str: saves to the given file (see plt.savefig())
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        if diff:
            tmp = ax.matshow(nabs(self.get_map(param, predict, uncert, ens=ens, uncer_type=uncer_type, tgt=False) -
                             self.get_map(param, predict, uncert, ens=ens, uncer_type=uncer_type, tgt=True)))
            ax.matshow(nabs(self.get_map(param, predict, uncert, ens=ens, uncer_type=uncer_type, tgt=False) -
                       self.get_map(param, predict, uncert, ens=ens, uncer_type=uncer_type, tgt=True)))
        else:
            if uncert:
                difference = nabs(self.get_map(param, predict, False, ens=ens, uncer_type=uncer_type, tgt=False) -
                                  self.get_map(param, predict, False, ens=ens, uncer_type=uncer_type, tgt=True))
                #tmp = ax.matshow(nabs(self.get_map(param, predict, False, ens=ens, uncer_type=uncer_type, tgt=False) -
                #                      self.get_map(param, predict, False, ens=ens, uncer_type=uncer_type, tgt=True)))
            else:
                #tmp = ax.matshow(self.get_map(param, predict, uncert=False, ens=ens, uncer_type=uncer_type, tgt=tgt))
                difference = self.get_map(param, predict, uncert=False, ens=ens, uncer_type=uncer_type, tgt=tgt)
            tmp = ax.matshow(self.get_map(param, predict, uncert, ens=ens, uncer_type=uncer_type, tgt=tgt),
                             vmin=difference.min(),
                             vmax=difference.max())
        fig.colorbar(tmp, ax=ax)
        ax.set_title(title, pad=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_facecolor('white')
        if save_fig and isinstance(save_fig, str):
            plt.savefig(fname=save_fig,
                        transparent=False,
                        facecolor='white')
        plt.show()


    def plot_parameter(self, param: str = 'T1',
                       predict: bool = False,
                       uncertainty: bool = False,
                       ens: bool = False,
                       uncer_type: str = "aleatoric",
                       tgt: bool = False,
                       diff: bool = False,
                       save_fig: str = False):
        """Creates a plot of the requested parameter and if wished of the uncertainty. The uncertainty is always multiplied with 10!

        :param param: "dB0", "B1", "T1", "T2", "b0_shift" will be considered as "dB0" and "b1_inhom" as "B1"
        :param predict: If true will first predict the dataset and then create the plot.
        :param uncertainty: if True also plots the uncertainty
        :param ens:  If False: The tensor contains the mean of all predictions and the mean of all the variances for that value. If True: The Tensor contains the mean of the prediction and the variance of the variances for that value.
        :param uncer_type: Default: "aleatoric", always used on pure GNLL, uncertainty is the mean of all uncertainties in the ensemble; "epistemic, uncertainty is the standard deviation of all uncertainties; "combined", adds up both uncertainties
        :param tgt: Default: False; If true returns the true targets as a matrix. Does not work if uncert=True. Needs self.data_type='phantom'.
        :param diff: Default: False; If true returns a plot of the difference between target and NN result. Does not work if uncert=True.
        :param save_fig: str: saves to the given file (see plt.savefig())
        """
        # prep title
        tmp_str_var = uncer_type[:4] + '. uncert.'

        if ens:
            tmp_str_ens = 'ensemble'
        else:
            tmp_str_ens = ''

        if tgt:
            tgt_str = 'tgt. '
        else:
            tgt_str = 'pred. '

        if diff:
            tgt_str = '|tgt. - pred.| '

        # convert param to printable name
        param_name = {'dB0': '${\Delta}$B$_0$ in [ppm]', 'B1': 'rel. B$_1$ in [%]',
                      'T1': 'T$_1$ in [s]', 'T2': 'T$_2$ in [s]'}[param]

        if uncertainty:
            self._plot_parameter(param=param,
                                 title='{} {} by NN {}'.format(param_name, tmp_str_var, tmp_str_ens),
                                 xlabel='Pixel',
                                 ylabel='Pixel',
                                 predict=predict,
                                 uncert=True,
                                 ens=ens,
                                 uncer_type=uncer_type,
                                 tgt=False,
                                 save_fig=save_fig)
        else:
            self._plot_parameter(param=param,
                                 title='{}{} by NN {}'.format(tgt_str, param_name, tmp_str_ens),
                                 xlabel='Pixel',
                                 ylabel='Pixel',
                                 predict=predict,
                                 ens=ens,
                                 tgt=tgt,
                                 diff=diff,
                                 save_fig=save_fig)

    def plot_lin_reg(self, param: str,
                     ax: plt.Axes = None,
                     save_fig: Union[bool, str] = False,
                     ens: bool = False,
                     uncer_type: str = "aleatoric",
                     **kwargs
                     ) -> plt.Axes:
        """Plots a scatter plot targets/predictions of the requested parameter and calculates R² and var(error).

        :param param: dB0 or b0_shift, B1 or b1_inhom, T1 or t1, T2 or t2
        :param ax: optionally hand an existing matplotlib.pyplot.Axes object for suplotting
        :param save_fig: True: saves as lin_reg.jpg to the working directory, str: saves to the given file (see plt.savefig())
        :param ens:  If False: The tensor contains the mean of all predictions and the mean of all the variances for that value. If True: The Tensor contains the mean of the prediction and the variance of the variances for that value.
        :param uncer_type: Default: "aleatoric", always used on pure GNLL, uncertainty is the mean of all uncertainties in the ensemble; "epistemic, uncertainty is the standard deviation of all uncertainties; "combined", adds up both uncertainties
        :param kwargs: kwargs for plt.savefig() (see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)
        :return: matplotlib.pyplot.Axes object
        """

        self.trainer[0].predictions_targets = self.trainer[0].predictions_targets.cpu()

        # set param
        net_labels = self.trainer[0].config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2'])
        if param not in net_labels and param in ['b0_shift', 'b1_inhom', 't1', 't2']:
            param = {'b0_shift': 'dB0', 'b1_inhom': 'B1', 't1': 'T1', 't2': 'T2'}[param]
        elif param in net_labels:
            pass
        else:
            raise ValueError('param has a value that cannot be interpreted: {}. It is possible the network'
                             ' is not trained for that parameter'.format(param))
        param_name = {'dB0': '${\Delta}$B$_0$', 'B1': 'rel. B$_1$', 'T1': 'T$_1$', 'T2': 'T$_2$'}[param]
        unit = {'dB0': 'ppm', 'B1': '%', 'T1': 's', 'T2': 's'}[param]

        # select
        for index, j in enumerate(net_labels):
            if param == j:
                # calculate the mean of all predictions in the ensemble
                pred = self.get_predictions(ens=ens, uncer_type=uncer_type).cpu()

                # linear regression
                lin_reg = linear_model.LinearRegression()
                lin_reg.fit(pred[:, index].reshape(-1, 1),
                            self.trainer[0].predictions_targets[:, index].reshape(-1, 1))
                lin_reg_pred = lin_reg.predict(pred[:, index].reshape(-1, 1))

                # variance
                var = (self.trainer[0].predictions_targets[:, index] - pred[:, index]).var()

                # R²
                r2 = self.r2(param=param)

                # get max and min for unit linear function
                x_max = float(pred[:, index].max())
                x_min = float(pred[:, index].min())

                # plot
                if ax is None:
                    fig, ax = plt.subplots()
                factor = 1
                if param == 'B1':
                    factor = 100
                ax.scatter(pred[:, index] * factor,
                           self.trainer[0].predictions_targets[:, index] * factor,
                           marker='.')
                ax.plot([x_min* factor, x_max * factor], [x_min * factor, x_max * factor], ls='-', color='k', label='f(x) = x')
                ax.plot(pred[:, index] * factor, lin_reg_pred[:, 0] * factor,
                        ls='--',
                        color='gray',
                        label='g(x)')
                ax.set_yticks(ax.get_xticks())
                ax.set_ylim(ax.get_xlim())
                ax.set_xlabel('predictions', size=11)
                ax.set_ylabel('targets', size=11)
                ax.set_title('{} [{}] with R² = {:.5f}, var(error) = {:.6f} \n '
                             'lin. reg. g(x) = {:.6f}*x + {:.6f}'.format(param_name, unit, r2, var, lin_reg.coef_[0, 0],
                                                               lin_reg.intercept_[0]), size=12)
                ax.grid('both')
                ax.legend(loc=4)

                # try to save the figure, but ensure it gets shown regardless
                try:
                    if save_fig and isinstance(save_fig, str):
                        plt.savefig(fname=save_fig,
                                    transparent=False,
                                    facecolor='white',
                                    **kwargs)
                    elif save_fig:
                        plt.savefig(fname='lin_reg.jpg',
                                    transparent=False,
                                    facecolor='white',
                                    **kwargs)
                except:
                    traceback.print_exc()
                    print("\033[0;30;41m Could not save the figure. If you have given a "
                          "path, is the ending set to something supported by plt.savefig()? \033[0m")

                self.trainer[0].predictions_targets = self.trainer[0].predictions_targets.to(self.data.dev)

                return ax

