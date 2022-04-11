# Standard library imports
import os
import time
from math import isnan
from typing import Optional, Union

# Third party imports
import matplotlib.pyplot as plt
from numpy import sum as nsum
from numpy import isin as nisin
from numpy import isnan as nisnan

from torch import load, save, cat, mean, abs, max, Tensor, zeros, exp, no_grad
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch import float32 as pytFl32

import tqdm

# our stuff
from .data import Data
from .neural_nets import DeepCEST_net, WASABI_net, WASABI_net2, DeepCEST_real_net, NeuralNet
from .loss import GNLL, alt_MSELoss, GNLLonSigma, GNLLonSigma_phys_m, GNLLonSigma_phys_m_new, GNLLonSigma_phys_m_alt, GNLLonSigma_phys_m_alt_noNorm, GNLLonSigma_phys_m_alt_useTgts, Phys_m_alt
from .auxiliary_functions import make_dir
from .auxiliary_functions import calc_print_b0shift, calc_print_b1, calc_print_t1, calc_print_t2


# trainings class
class Trainer:
    """Class to simplify training of the Model."""
    def __init__(self,
                 data: Data,
                 use_tb: bool = False):
        """Class to simplify training of the Model.
        :param data: data class object
        :param use_tb: using tensorboard a few more statistics will be captured, requires tensorboard to be installed
        """
        self.data = data  # data class object, contains the DataLoader
        self.config = data.config  # config file
        self.loss_train = []  # list of tuple (epoch, average loss per trainings epoch)
        self.loss_val = []  # list of tuple (epoch, average validation loss)
        self.train_loader = data.train_loader  # DataLoader of the trainings data
        self.val_loader = data.val_loader  # DataLoader of the val data
        self.test_loader = data.test_loader  # DataLoader of the test data
        self.predictions = None  # filled after eval with the predictions for the val data
        self.predictions_targets = None  # filled after eval with the prediction targets for the val data
        self.epochCount = 0  # total epochs run
        self.dirpath_save = self.config['OUTPUT_DIR']  # directory for autosaves
        # saves the model and aborts training when possible over-fitting is detected
        self.abort_on_overfit = self.config.get('ABORT_ON_OVERFIT', False)  # if true aborts model on over-fit
        self.val_freq = self.config.get('VAL_FREQ', 2)  # every x epochs the validation set gets checked
        self.autosave_freq = self.config.get('AUTOSAVE_FREQ', 10)  # every n epochs the model autosaves
        self.reg_save = self.config.get('REG_SAVE', False)  # saves every n epochs into a separat file
        self.time_trained = 0.  # seconds spend training
        self.time_start = None  # system time when the last time was taken
        self.overfit_check = (0, 0.)  # (epochs since assigned, lowest val_error)
        self.n_overfit_epochs = self.config.get('ABORT_AFTER_N_OVERFIT_EPOCHS', 10)  # ABORT_AFTER_N_OVERFIT_EPOCHS
        self.opt = None  # the optimizer
        self.use_tb = use_tb  # the flag on whether tensorboard should be used
        self.physM = False  # use physical model toggle
        self.monteCarloDropout = False

        # TODO temporary deprecation warning
        if self.config.get('LEARNING_RATE'):
            print('\033[0;30;42m The learning rate as a config parameter has been removed. Please add it to '
                  'the kwargs of the optimizer. See nn.module() for that. \033[0m')

        # monteCarloDropout if True, will not set the net into eval() mode during prediction
        if self.config.get('MONTECARLODROPOUT', False):
            if isinstance(self.config.get('MONTECARLODROPOUT'), bool):
                self.monteCarloDropout = self.config.get('MONTECARLODROPOUT')
            else:
                print('\033[0;30;41m The config parameter MonteCarloDropout was '
                      'not boolean and has been set to False. \033[0m')

        # fix for false learning rate in old configs (was orig. 0.0001 became then 0.001)
        # configs pre 2021.04.15
        if isinstance(self.config.get('OPTIMIZER'), str):
            if self.config.get('OPTIMIZER', 'no') == 'Adam':
                self.config['OPTIMIZER'] = {1: 'Adam', '1_kwargs': {'lr': 0.0001}}

        # check for correct datatype
        if self.config.get('ABORT_ON_LOSS') is not None:
            try:
                if isinstance(float(self.config.get('ABORT_ON_LOSS')), float):
                    self.config['ABORT_ON_LOSS'] = float(self.config.get('ABORT_ON_LOSS'))
            except:
                print('Your ABORT_ON_LOSS parameter was not convertible to a float and has been set to a default '
                      'value of -11')
                self.config['ABORT_ON_LOSS'] = -11.

        if self.config.get('ABORT_ON_TIME') is not None:
            try:
                if isinstance(float(self.config.get('ABORT_ON_TIME')), float):
                    self.config['ABORT_ON_TIME'] = float(self.config.get('ABORT_ON_TIME'))
            except:
                print('Your ABORT_ON_TIME parameter was not convertible to a float and has been set to a default '
                      'value of 604800s (1 week)')
                self.config['ABORT_ON_TIME'] = 604800.

        if self.config.get('ABORT_ON_SLOW_TRAIN') is not None:
            if isinstance(self.config.get('ABORT_ON_SLOW_TRAIN'), bool):
                pass
            else:
                # check for correct type
                assert isinstance(self.config.get('ABORT_ON_SLOW_TRAIN'), float), "The ABORT_ON_SLOW_TRAIN " \
                                                                              "value in the config is not" \
                                                                              " a float type."

        if isinstance(self.reg_save, int):
            pass
        elif isinstance(self.reg_save, bool):
            # handing over a default value if REG_SAVE is True even if that is not a valid option
            if self.reg_save:
                self.reg_save = 10
        elif isinstance(self.reg_save, float):
            self.reg_save = int(self.reg_save)
        else:
            raise ValueError('The REG_SAVE parameter has to be a integer or false. '
                             'REG_SAVE: {} is of type {}.'.format(self.reg_save, type(self.reg_save)))

        # choose torch.nn.Module type neural net to be trained
        if self.config['NET'] == 'DeepCEST':
            self.net = DeepCEST_net().to(self.data.dev)
        elif self.config['NET'] == 'DeepCEST_real':
            self.net = DeepCEST_real_net().to(self.data.dev)
        elif self.config['NET'] == 'WASABI':
            self.net = WASABI_net().to(self.data.dev)
        elif self.config['NET'] == 'WASABI2':
            self.net = WASABI_net2().to(self.data.dev)
        elif self.config['NET'] == 'CUSTOM':
            if self.config.get('LAYER_KWARGS') is not None:
                self.net = NeuralNet(self.config['LAYERS'],
                                     self.config['N_NEURONS'],
                                     self.config['LAYER_KWARGS']).to(self.data.dev)
            else:
                self.net = NeuralNet(self.config['LAYERS'],
                                     self.config['N_NEURONS'],
                                     {}).to(self.data.dev)
        else:
            raise ValueError('no valid NN found.')

        # choose pytorch type loss function, or with physical model
        if isinstance(self.config['LOSS_FKT'], dict):
            if self.config['LOSS_FKT']['FCT'] == 'MSELoss':
                # initialized MSELoss with kwargs
                self.crit = alt_MSELoss(**self.config['LOSS_FKT']['kwargs'])
            elif self.config.get('LOSS_FKT')['FCT'] == 'GNLLonSigma_PhysM':
                # initialized gaussian log-likelihood with training on sigma and the physical model
                self.crit = GNLLonSigma_phys_m(n=self.data.n_tgt_params,
                                               dev=self.data.dev,
                                               param=self.config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2']),
                                               **self.config['LOSS_FKT']['kwargs'])
                self.physM = True
                self.lambda_factor = self.config['LOSS_FKT']['kwargs']['lambda_fact']
            # phyM including relaxation during dead_time (30 µs) and spoiler (5.5 ms)
            elif self.config.get('LOSS_FKT')['FCT'] == 'GNLLonSigma_PhysM_alt':
                # initialized gaussian log-likelihood with training on sigma and the physical model
                self.crit = GNLLonSigma_phys_m_alt(n=self.data.n_tgt_params,
                                                   dev=self.data.dev,
                                                   param=self.config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2']),
                                                   **self.config['LOSS_FKT']['kwargs'])
                self.physM = True
                self.lambda_factor = self.config['LOSS_FKT']['kwargs']['lambda_fact']
            # like GNLLonSigma_phys_m_alt but norms will be undone before the crit will be calculated
            elif self.config.get('LOSS_FKT')['FCT'] == 'GNLLonSigma_PhysM_alt_noNorm':
                # initialized gaussian log-likelihood with training on sigma and the physical model
                self.crit = GNLLonSigma_phys_m_alt_noNorm(n=self.data.n_tgt_params,
                                                          dev=self.data.dev,
                                                          param=self.config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2']),
                                                          **self.config['LOSS_FKT']['kwargs'])
                self.physM = True
                self.lambda_factor = self.config['LOSS_FKT']['kwargs']['lambda_fact']
            # like GNLLonSigma_phys_m_alt but norms will be undone before the crit will be calculated
            elif self.config.get('LOSS_FKT')['FCT'] == 'GNLLonSigma_PhysM_alt_useTgts':
                # initialized gaussian log-likelihood with training on sigma and the physical model
                self.crit = GNLLonSigma_phys_m_alt_useTgts(n=self.data.n_tgt_params,
                                                           dev=self.data.dev,
                                                           param=self.config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2']),
                                                           **self.config['LOSS_FKT']['kwargs'])
                self.physM = True
                self.lambda_factor = self.config['LOSS_FKT']['kwargs']['lambda_fact']
            # like GNLLonSigma_phys_m_alt but the GNLL term gets set to zero before returning
            elif self.config.get('LOSS_FKT')['FCT'] == 'PhysM_alt':
                # initialized training on the physical model
                self.crit = Phys_m_alt(n=self.data.n_tgt_params,
                                       dev=self.data.dev,
                                       param=self.config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2']),
                                       **self.config['LOSS_FKT']['kwargs'])
                self.physM = True
                self.lambda_factor = self.config['LOSS_FKT']['kwargs']['lambda_fact']
            # FUNCTION itself is instable
            elif self.config.get('LOSS_FKT')['FCT'] == 'GNLLonSigma_PhysM_new':
                # initialized gaussian log-likelihood with training on sigma and the physical model
                self.crit = GNLLonSigma_phys_m_new(n=self.data.n_tgt_params,
                                                   dev=self.data.dev,
                                                   param=self.config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2']),
                                                   **self.config['LOSS_FKT']['kwargs'])
                self.physM = True
                self.lambda_factor = self.config['LOSS_FKT']['kwargs']['lambda_fact']
            else:
                raise ValueError('No valid dict loss function was found. Please refer to the net_config_explanation.yaml.')
        elif self.config['LOSS_FKT'] == 'MSELoss':
            # initialized MSELoss
            self.crit = alt_MSELoss()
        elif self.config['LOSS_FKT'] == 'GNLL':
            # initialized gaussian log-likelihood with training on log(sigma), as deepCEST paper
            self.crit = GNLL(self.data.n_tgt_params, self.data.dev)
        elif self.config.get('LOSS_FKT') == 'GNLLonSigma':
            # initialized gaussian log-likelihood with training on sigma
            self.crit = GNLLonSigma(self.data.n_tgt_params, self.data.dev)
        else:
            raise ValueError('No valid loss function was found. Please refer to the net_config_explanation.yaml.')

        # initialize Tensorboard
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter

                # logging location
                self.writer = SummaryWriter(os.path.join(self.dirpath_save, 'tensorboard'))
                # create custom charts for tensorboard
                tb_layout = {'loss': {'loss comparison': ['Multiline', ['trainings loss', 'validation loss',
                                                                        'trainings loss mean']]},
                             'mean': {'mean_rel comparison': ['Multiline', ['mean_rel_db0', 'mean_rel_b1',
                                                                            'mean_rel_t1', 'mean_rel_t2']]},
                             'T1 & T2 comparison': {'mean_abs': ['Multiline', ['mean_abs_t1', 'mean_abs_t2']],
                                                    'max_abs': ['Multiline', ['max_abs_t1', 'max_abs_t2']],
                                                    'mean_rel': ['Multiline', ['mean_rel_t1', 'mean_rel_t2']]},
                             'physical model val error': {'loss_types': ['Multiline',
                                                                         ['GNLL_loss_mean',
                                                                          'GNLL_loss_sum',
                                                                          'phys_m_loss']]}
                             }
                self.writer.add_custom_scalars(layout=tb_layout)

                # add NN structure to tensorboard
                tmp_img = next(iter(self.test_loader))[0][0]
                self.writer.add_graph(model=self.net, input_to_model=tmp_img)

            except ModuleNotFoundError:
                self.use_tb = False
                print('\033[0;30;41m Tensorboard is not installed. \033[0m')

    def _choose_opt(self, opt_name: str, **opt_kwargs):
        """Choose the torch.optim.Optimizer. Does NOT consider the datatype in the config which can lead to errors
        so best use _set_opt().
        """
        if opt_name == 'Adam':
            # initialized ADAM
            self.opt = Adam(self.net.parameters(), **opt_kwargs)
        elif opt_name == 'SGD':
            # initialized stochastic gradient decent
            self.opt = SGD(self.net.parameters(), **opt_kwargs)
        else:
            raise Exception('No valid optimizer found.')

    def _set_opt(self, override: Union[bool, int] = False, optimizer: Union[bool, str, dict] = False):
        """Chooses the torch.optim.Optimizer based on the epoch count and input given in the config.
        Epoch count starts at one.

        :param override: used for loading a saved model
        :param optimizer: False: uses the config optimizer, str or dict: overrides the config optimizer (same format)
        """
        epochCount = self.epochCount
        # when loading the epoch count counts through all epochs to get the correct optimizer
        if override:
            epochCount = override

        if not optimizer:
            optimizer = self.config['OPTIMIZER']

        # if only a string is given, so only one optimizer
        if isinstance(optimizer, str):
            self._choose_opt(optimizer)
        # if the given value is a dictionary assume multiple optimizer depending on the epoch
        # start with epoch 1
        elif isinstance(optimizer, dict):
            if optimizer.get(epochCount, False):
                # check for kwargs
                if optimizer.get(str(epochCount)+'_kwargs', False):
                    # handover with kwargs
                    self._choose_opt(optimizer.get(epochCount),
                                     **optimizer.get(str(epochCount)+'_kwargs'))
                else:
                    # if there are no kwargs
                    self._choose_opt(optimizer.get(epochCount))
        else:
            raise Exception('Could not set a optimizer. The OPTIMIZER parameter in the config possibly has the '
                            'wrong format. See the net_config_explanation.yaml file.')

    def _load_opt(self, optimizer: Union[bool, str, dict] = False):
        """When loading a model the optimizer is choose based on the current epochCount within that model
        and the config file.
        :param optimizer: False: uses the config optimizer, str or dict: overrides the config optimizer (same format)
        """
        # inelegant, but since it is only done once..
        for i in range(self.epochCount):
            self._set_opt(override=i, optimizer=optimizer)

    def save_net(self, filepath: str):
        """Saves the state_dict from the net and optimizer.
        Also saves the epochCount and the loss_train.
        :param filepath: file path of the save file
        """
        make_dir(os.path.split(filepath)[0])

        # everything is saved as a dictionary
        save({'net': self.net.state_dict(),
              'optimizer': self.opt.state_dict(),
              'loss_train': self.loss_train,
              'loss_val': self.loss_val,
              'epochCount': self.epochCount,
              'config': self.config,
              'time_trained': self.time_trained,
              'overfit_check_tuple': self.overfit_check
              }, filepath)

    def load_net(self, filepath: str):
        """Loads the saved state_dict into the net and optimizer.
        Also loads the epochCount and the loss_train.
        :param filepath: file path of the load file
        """
        # load binary file into dictionary
        file_dict = load(filepath)

        # everything is loaded from the dictionary
        self.net.load_state_dict(file_dict['net'])
        self.epochCount = file_dict['epochCount']
        self.loss_train = file_dict['loss_train']
        self.loss_val = file_dict['loss_val']
        self.time_trained = file_dict['time_trained']
        self.overfit_check = file_dict['overfit_check_tuple']
        self._load_opt(file_dict['config'].get('OPTIMIZER'))
        self.opt.load_state_dict(file_dict['optimizer'])
        if self.epochCount > 1:
            self.data._change_train_loader_batchsize(self.config.get('BATCH_SIZE'),
                                                     self.epochCount)
            self.train_loader = self.data.train_loader
            

        # check and print if the configs are different
        if self.config != file_dict['config']:
            print('Warning! Be aware that the two config files are different!')
            # print('Config of the saved net:\n', file_dict['config'])
            # print('Config currently used:\n', self.config)

    def plot_loss_save(self):
        """Creates a plot of the current loss_train and loss_val and saves them as current_loss.png.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('running loss of each epoch')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.plot([i for i, _ in self.loss_train],
                [j for _, j in self.loss_train],
                ls='-', label='loss_train')
        ax.plot([i for i, _ in self.loss_val],
                [j for _, j in self.loss_val],
                ls='--', label='loss_val')
        ax.grid('both')
        ax.legend()
        make_dir(self.dirpath_save)
        fig.savefig(os.path.join(self.dirpath_save,
                                 self.config['CONFIG_NAME'] +
                                 '_current_loss.png'),
                    transparent=False,
                    facecolor='white')
        plt.close()

    def _plot_weights_and_biases_to_tb(self):
        """Plots the weights and biases with labeled heat map.

        """
        state_dict = self.net.state_dict()

        for i in state_dict.keys():
            if len(state_dict[i].shape) == 2:
                fig = plt.figure(figsize=(6, 10))
                plt.imshow(state_dict[i].cpu())
            elif len(state_dict[i].shape) == 1:
                fig = plt.figure(figsize=(10, 2))
                plt.imshow(state_dict[i].reshape(1, -1).cpu())
            else:
                fig = plt.figure()

            plt.title(i)
            plt.xlabel('max() = ' + str(state_dict[i].max()) +
                       ' min() = ' + str(state_dict[i].min()) +
                       '\n # nans = ' + str(nsum(nisnan(state_dict[i].cpu()).numpy())) +
                       ' # zeros = ' + str(nsum(nisin(state_dict[i].cpu(), 0.))))
            plt.colorbar(fraction=0.046)

            self.writer.add_figure('{}'.format(i), fig, global_step=self.epochCount, close=True)
            self.writer.close()

    def _plot_loss_to_tb(self,
                         GNLL_loss_mean: float,
                         GNLL_loss_sum: float,
                         phys_m_loss: float):
        """Updates the writer with the different loss values in the physical model to find a balance between it and GNLL.

        :param GNLL_loss_mean:
        :param GNLL_loss_sum:
        :param phys_m_loss:
        """
        self.writer.add_scalar(tag='GNLL_loss_mean', scalar_value=GNLL_loss_mean, global_step=self.epochCount)
        self.writer.add_scalar(tag='GNLL_loss_sum', scalar_value=GNLL_loss_sum, global_step=self.epochCount)
        self.writer.add_scalar(tag='phys_m_loss', scalar_value=phys_m_loss, global_step=self.epochCount)
        self.writer.close()

    def _core_loop(self, data_loader: DataLoader or tqdm,
                   loader_type: str) -> (float, float):
        """Core loop over the dataset of the training/evaluation/validation.
        :param data_loader: The data loader which is to be iterated over
        :param loader_type: what type of data loader it is: 'train', 'test', 'val'
        :return: returns n_samples the total number of individual samples; the cumulative loss of all individual samples added up; the cumulative mean of each batch
        Additionally if the loader_type is set to test it also sets the predictions and predictions_tgts variable
        """
        if not loader_type == 'test' and not loader_type == 'train' and not loader_type == 'val':
            raise NameError('You have passed the wrong label to the core loop. Please use train, val or test.')

        cuml_loss_sum_epoch = 0  # cumulative loss of the test dataset
        cuml_loss_mean_epoch = 0  # cumulative loss of the test dataset uncorrected for batch size
        n_samples = 0  # set counter for sample size
        output_concat = []  # list of calculated targets
        target_concat = []  # list of actual targets

        # prep phys model logging
        if self.physM and self.use_tb:
            cuml_gnll_loss_sum = 0.
            cuml_gnll_loss_mean = 0.
            cuml_phys_m_loss = 0.

        # set net state
        if loader_type == 'train' or self.monteCarloDropout:
            self.net.train()
        else:
            self.net.eval()

        for input_x, input_y, input_x_no_noise, y_no_noise_all in data_loader:
            if loader_type == 'train':
                self.opt.zero_grad()  # zero the gradients
            output = self.net(input_x)  # forward through the net

            if self.physM:
                # calculate the loss with physical model; phys_m_loss is the mean physical loss of the batch
                if self.config.get('LOSS_FKT')['FCT'] == 'GNLLonSigma_PhysM_alt_useTgts':
                    gnll_loss, phys_m_loss = self.crit(output, input_y, input_x_no_noise, y_no_noise_all, self.data)
                else:
                    gnll_loss, phys_m_loss = self.crit(output, input_y, input_x_no_noise, self.data)

                # have to achieve the following while keeping the GNLL loss and phys mod loss separate for tb
                loss = gnll_loss.mean() + phys_m_loss * self.lambda_factor
                loss_sum = gnll_loss.sum() + phys_m_loss * len(input_x) * self.lambda_factor
            else:
                # calculate the loss without physical model
                loss, loss_sum = self.crit(output, input_y)  # calculate the loss

            # log: phys model specific loss logging
            if self.use_tb and self.physM and loader_type == 'val':
                cuml_gnll_loss_mean += gnll_loss.mean().item()
                cuml_gnll_loss_sum += gnll_loss.sum().item()
                cuml_phys_m_loss += phys_m_loss.item()

            n_samples += len(input_x)  # add up the number of samples
            cuml_loss_mean_epoch += loss.item()  # add loss uncorrected for batch size
            cuml_loss_sum_epoch += loss_sum.item()  # add batch loss

            # back step
            if loader_type == 'train':
                loss.backward()  # update the gradients
                self.opt.step()  # one optimizer step
            if loader_type == 'test' or loader_type == 'val':
                # make list of batch results and batch targets in correct order
                output_concat.append(output)
                target_concat.append(input_y)

        # log: loggin different loss to tb
        if self.use_tb and self.physM and loader_type == 'val':
            self._plot_loss_to_tb(cuml_gnll_loss_mean / len(data_loader),
                                  cuml_gnll_loss_sum / n_samples,
                                  cuml_phys_m_loss / len(data_loader))

        # set predictions;
        if loader_type == 'test' or loader_type == 'val':
            # concatenate all output tensors
            self.predictions = cat(output_concat)
            self.predictions_targets = cat(target_concat)

            # possibly undo norms
            if self.data.norm_tgts_bounds is not None and self.data.norm_tgts_bounds:
                self.predictions, self.predictions_targets = self.data.undo_norm_tgts(self.predictions,
                                                                                      self.predictions_targets)

            if self.config.get('LOSS_FKT') == 'GNLL':
                # changes the uncertainty values from log(std) to std
                self.predictions[:, self.data.n_tgt_params:] = exp(self.predictions[:, self.data.n_tgt_params:])
            elif self.config.get('LOSS_FKT') == 'GNLLonSigma':
                # changes the uncertainty values from std to abs(std)
                self.predictions[:, self.data.n_tgt_params:] = abs(self.predictions[:, self.data.n_tgt_params:])

        return n_samples, cuml_loss_sum_epoch, cuml_loss_mean_epoch

    def _validation(self):
        """Calculates the loss of the validation set and appends that to loss_val.
        """
        # turn gradients off
        with no_grad():
            n_samples, cuml_loss_sum_epoch, _ = self._core_loop(data_loader=self.val_loader, loader_type='val')

        # append average loss of the epoch to self.loss_val
        self.loss_val.append((self.epochCount, cuml_loss_sum_epoch / n_samples))

    def _update_time(self):
        """Updates the time spend training and afterwards sets the time_start variable to the current time.
        """
        self.time_trained += time.time() - self.time_start
        self.time_start = time.time()

    def _abort_on_slow_train(self) -> bool:
        """checks if the mean trainings error decrease speed is slow and returns true if so.
        """
        # trainings speed check based on mean error decrease of the trainings set
        if self.config.get('ABORT_ON_SLOW_TRAIN', False):
            if len(self.loss_train) > 20:
                # calc the mean loss over 10 epochs
                current_mean_loss = sum([j for i, j in self.loss_train[-10:]]) / 10.
                past_mean_loss = sum([j for i, j in self.loss_train[-20:-10]]) / 10.

                if past_mean_loss - current_mean_loss < self.config.get('ABORT_ON_SLOW_TRAIN'):
                    print('The decrease of the 10 epochs mean trainings error over the last 10 epochs was less then '
                          '{} so the training was stopped.'.format(past_mean_loss - current_mean_loss))
                    # if loss decrease to small on the train set, abort training
                    return True
        return False

    def _abort_on_loss(self) -> bool:
        """Checks if the loss is below a given value and returns true if that is the case. Otherwise returns false.
        """
        if self.config.get('ABORT_ON_LOSS', False):
            if float(self.config.get('ABORT_ON_LOSS')) > self.loss_train[-1][1]:
                # autosave
                self._update_time()

                self.save_net(os.path.join(self.dirpath_save,
                                           self.config['CONFIG_NAME'] +
                                           '_model_autosave.pt'))

                print('Required loss ({:.5f}) was reached ({:.5f}), training was stopped.'.format(
                    self.config.get('ABORT_ON_LOSS'), self.loss_train[-1][1]))
                return True
        return False

    def _abort_on_time(self) -> bool:
        """Checks if the training lasted longer than a given time and returns True if that is the case. Otherwise False.
        """
        if self.config.get('ABORT_ON_TIME', False):
            self._update_time()

            if float(self.config.get('ABORT_ON_TIME')) < self.time_trained:
                # autosave
                self.save_net(os.path.join(self.dirpath_save,
                                           self.config['CONFIG_NAME'] +
                                           '_model_autosave.pt'))

                print('Required time of {:.4f} h was reached (trained {:.4f} h), training was stopped.'.format(
                    float(self.config.get('ABORT_ON_TIME')) / 3600.,
                    self.time_trained / 3600.))
                return True
        return False

    def _overfit_check_and_save(self) -> bool:
        """Checks if over fitting happens and saves if the val_loss is lower than before. If over fitting happens, it
        returns True, otherwise False.
        """
        # save best epoch and check for over-fitting and abort if configured that way
        if self.overfit_check[1] > self.loss_val[-1][1]:  # better val_loss
            self.overfit_check = (0, self.loss_val[-1][1])

            # add time trained
            self._update_time()

            # save best current model
            self.save_net(os.path.join(self.dirpath_save,
                                       self.config['CONFIG_NAME'] +
                                       '_best_model_save.pt'))
        elif self.overfit_check[0] < self.n_overfit_epochs:  # worse val_loss
            self.overfit_check = (self.overfit_check[0] + 1, self.overfit_check[1])
        elif self.abort_on_overfit:
            print('Abort due to likelihood of overfit.')
            return True
        else:
            return False

    def _update_tb_training(self):
        """When further logging with tensorboard is enabled this function updates all values that would get calculated
        in 'parameter_min_max_check' in eval_wasabi.py to tensorboard.
        """
        for i, j in enumerate(self.config.get('TYPE_PARAMS', ['dB0', 'B1', 'T1', 'T2'])):
            if j == 'dB0':
                mean_abs_db0, max_abs_db0, mean_rel_db0 = calc_print_b0shift(self.predictions,
                                                                             self.predictions_targets,
                                                                             index=i)
                self.writer.add_scalar(tag='mean_abs_db0', scalar_value=mean_abs_db0, global_step=self.epochCount)
                self.writer.add_scalar(tag='max_abs_db0', scalar_value=max_abs_db0, global_step=self.epochCount)
                self.writer.add_scalar(tag='mean_rel_db0', scalar_value=mean_rel_db0, global_step=self.epochCount)
                if self.predictions.shape[1] > self.predictions_targets.shape[1]:
                    self.writer.add_scalar(tag='dB0_uncert',
                                           scalar_value=self.predictions[:, i + self.data.n_tgt_params].mean(),
                                           global_step=self.epochCount)
            elif j == 'B1':
                mean_abs_b1, max_abs_b1, mean_rel_b1 = calc_print_b1(self.predictions,
                                                                     self.predictions_targets,
                                                                     index=i)
                self.writer.add_scalar(tag='mean_abs_b1', scalar_value=mean_abs_b1, global_step=self.epochCount)
                self.writer.add_scalar(tag='max_abs_b1', scalar_value=max_abs_b1, global_step=self.epochCount)
                self.writer.add_scalar(tag='mean_rel_b1', scalar_value=mean_rel_b1, global_step=self.epochCount)
                if self.predictions.shape[1] > self.predictions_targets.shape[1]:
                    self.writer.add_scalar(tag='B1_uncert',
                                           scalar_value=self.predictions[:, i + self.data.n_tgt_params].mean(),
                                           global_step=self.epochCount)
            elif j == 'T1':
                mean_abs_t1, max_abs_t1, mean_rel_t1 = calc_print_t1(self.predictions,
                                                                     self.predictions_targets,
                                                                     index=i)
                self.writer.add_scalar(tag='mean_abs_t1', scalar_value=mean_abs_t1, global_step=self.epochCount)
                self.writer.add_scalar(tag='max_abs_t1', scalar_value=max_abs_t1, global_step=self.epochCount)
                self.writer.add_scalar(tag='mean_rel_t1', scalar_value=mean_rel_t1, global_step=self.epochCount)
                if self.predictions.shape[1] > self.predictions_targets.shape[1]:
                    self.writer.add_scalar(tag='T1_uncert',
                                           scalar_value=self.predictions[:, i + self.data.n_tgt_params].mean(),
                                           global_step=self.epochCount)
            elif j == 'T2':
                mean_abs_t2, max_abs_t2, mean_rel_t2 = calc_print_t2(self.predictions,
                                                                     self.predictions_targets,
                                                                     index=i)
                self.writer.add_scalar(tag='mean_abs_t2', scalar_value=mean_abs_t2, global_step=self.epochCount)
                self.writer.add_scalar(tag='max_abs_t2', scalar_value=max_abs_t2, global_step=self.epochCount)
                self.writer.add_scalar(tag='mean_rel_t2', scalar_value=mean_rel_t2, global_step=self.epochCount)
                if self.predictions.shape[1] > self.predictions_targets.shape[1]:
                    self.writer.add_scalar(tag='T2_uncert',
                                           scalar_value=self.predictions[:, i + self.data.n_tgt_params].mean(),
                                           global_step=self.epochCount)
        self.writer.close()

    def train(self, n_epochs: int):
        """Trains the neural net for the number of given epochs.
        :param n_epochs: number of epochs to be trained
        """
        self.time_start = time.time()
        # if only evaluation data is loaded, raise error
        if self.data._evaluation:
            raise Exception('You cannot train with a dataset that is loaded only for evaluation.')

        # on first training necessary for postfix
        if not self.loss_val:
            self._validation()
            self.overfit_check = (0, self.loss_val[0][1])

        # create loop iterator to enable live viewing in the command line
        loop_iterator = tqdm.tqdm(range(n_epochs), ascii=True, position=0)

        # set batch size for integer
        if isinstance(self.config.get('BATCH_SIZE'), int):
            self.data._change_train_loader_batchsize(self.config.get('BATCH_SIZE'),
                                                     self.epochCount)
            self.train_loader = self.data.train_loader

        for _ in loop_iterator:
            self.epochCount += 1

            # set batch size for dictionaries
            if isinstance(self.config.get('BATCH_SIZE'), dict):
                if self.config.get('BATCH_SIZE').get(self.epochCount) is not None:
                    self.data._change_train_loader_batchsize(self.config.get('BATCH_SIZE'),
                                                             self.epochCount)
                    self.train_loader = self.data.train_loader

            # checks config for possible optimizer change this epoch
            self._set_opt()

            # core loop for loss
            n_samples, cuml_loss_sum_epoch, cuml_loss_mean_epoch = self._core_loop(data_loader=self.train_loader,
                                                                                   loader_type='train')

            # append average loss of the epoch
            self.loss_train.append((self.epochCount, cuml_loss_sum_epoch / n_samples))

            # ABORT if the loss is nan
            if isnan(self.loss_train[-1][1]) and self.config.get('ABORT_ON_NAN', True):
                print('ABORT: The trainings loss is nan.')
                break

            # save model every couple epochs
            if self.epochCount % self.autosave_freq == 0:
                # add time trained
                self._update_time()

                self.save_net(os.path.join(self.dirpath_save,
                                           self.config['CONFIG_NAME'] +
                                           '_model_autosave.pt'))

            # save model into separate file if specified
            if self.reg_save and self.epochCount % self.reg_save == 0:
                # add time trained
                self._update_time()

                self.save_net(os.path.join(self.dirpath_save,
                                           self.config['CONFIG_NAME'] +
                                           '_model_epoch_{}.pt'.format(self.epochCount)))

            if self.epochCount % self.val_freq == 0:
                # predict the validation dataset
                self._validation()

                # update tensorboard
                if self.use_tb:
                    self._update_tb_training()
                    self.writer.add_scalar(tag='validation loss',
                                           scalar_value=self.loss_val[-1][1],
                                           global_step=self.epochCount)
                    self.writer.add_scalar(tag='trainings loss mean',
                                           scalar_value=cuml_loss_mean_epoch/len(self.train_loader),
                                           global_step=self.epochCount)
                    self.writer.close()

                # save plot of current loss
                # should the file still be open or not otherwise not accessible
                try:
                    self.plot_loss_save()
                except:
                    time.sleep(1)  # added to avoid access conflicts on really tiny datasets
                    self.plot_loss_save()

                # save best epoch and check for over-fitting and ABORT if configured that way
                if self._overfit_check_and_save():
                    break

            # update tensorboard
            if self.use_tb:
                self.writer.add_scalar(tag='trainings loss',
                                       scalar_value=self.loss_train[-1][1],
                                       global_step=self.epochCount)
                # not needed ATM
                # self._plot_weights_and_biases_to_tb()
                self.writer.close()

            # postfix the training bar
            postfix = [('sum loss', self.loss_train[-1][1]),
                       ('val sum loss', self.loss_val[-1][1]),
                       ('epoch', self.epochCount)]  # prepare the progress bar
            loop_iterator.set_postfix(postfix)  # update the progress bar

            # ABORT based on mean error decrease of the trainings set
            if self._abort_on_slow_train():
                break

            # ABORT training if it lasted longer that a given value
            if self._abort_on_time():
                break

            # ABORT training if the loss is below a given value
            if self._abort_on_loss():
                break

        self.time_trained += time.time() - self.time_start  # add time trained
