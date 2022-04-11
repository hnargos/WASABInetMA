from torch import nn, sum, div, mul, log, tensor, sub, exp, square, device, log, sqrt, ones_like
from math import pi
from typing import Optional, Union
from torch import Tensor, tensor, ones
from .auxiliary_functions import wasabiti, wasabiti_new, wasabiti_alt
from torch import float32 as pytFl32
from .data import Data

class Phys_m_alt(nn.Module):
    """Calculates the lambda * pyhsical model (including relaxation during dead_time
    (30 µs) and spoiler (5.5 ms)) MSE. Trains on uncertainties = sigma.
    Basically GNLLonSigma_phys_m_alt with the GNLL return set to 0.

    It also takes the abs of sigma to avoid problems with the log(sigma). This is also done for the prediction
    """
    def __init__(self, n: Tensor,
                 dev: device,
                 param: list,
                 x: Union[Tensor, list],
                 trec: Union[Tensor, list, float],
                 lambda_fact: Tensor = 1.,
                 b0_shift: Tensor = 0.,
                 b1: Tensor = 3.75,
                 t1: Tensor = 2.,
                 t2: Tensor = 0.1,
                 tp: Tensor = 0.005,
                 gamma: Tensor = 42.5764):
        """

        :param n: number of tgt parameter
        :param dev: device
        :param param:
        :param lambda_fact:
        :param x: frequency offsets [ppm]
        :param trec: recover time between different offsets/measurements. Can be a float or an array [s]
        :param b0_shift: b0_shift-shift [ppm]
        :param b1: B1 peak amplitude [µT]
        :param t1: longitudinal relaxation time (T_1) [s]
        :param t2: transversal relaxation time (T_2) [s]
        :param tp: duration of the WASABI pulse [s]
        :param gamma: gyromagnetic ratio [MHz/T]
        """
        super().__init__()
        self.dev = dev
        self.third_term_const = 0.5 * log(tensor(2 * pi, device=dev))
        self.n = n  # number of parameters

        # physical model params
        self.offsets = tensor(x, dtype=pytFl32, device=dev)
        self.trec = tensor(trec, dtype=pytFl32, device=dev)
        self.b0_shift = tensor([b0_shift], dtype=pytFl32, device=dev)
        self.b1 = tensor([b1], dtype=pytFl32, device=dev)
        self.t1 = tensor([t1], dtype=pytFl32, device=dev)
        self.t2 = tensor([t2], dtype=pytFl32, device=dev)
        self.gamma = tensor(gamma, dtype=pytFl32, device=dev)
        self.freq = tensor(gamma * 3., dtype=pytFl32, device=dev)
        self.tp = tensor(tp, dtype=pytFl32, device=dev)

        # scaling factor of the physical model
        self.lambda_fact = tensor(lambda_fact, dtype=pytFl32, device=dev)
        # parameter list
        self.param = param
        self.param_missing = list({'dB0', 'B1', 'T1', 'T2'} - set(self.param))


        # batch size
        self.batch_size = 1
        # base return size, used to bring parameters to correct shape
        self.ret_shape = ones([31], dtype=pytFl32, device=dev)


        # initialize MSE for the physical model
        self.mse_mean = nn.MSELoss(reduction='mean')
        self.mse_sum = nn.MSELoss(reduction='sum')

    def forward(self,
                y_calc: Optional[Tensor],
                y_tgt: Optional[Tensor],
                input_x: Tensor,
                data: Data) -> [Tensor, Tensor]:
        """Calculates the gaussian log-likelihood loss.

        :param y_calc: mini-batch containing the features and uncertainties (in log(sigma) form)
        :param y_tgt: mini-batch containing the targets
        :param input_x: the z-spectra of that trainings batch
        :param data: the data object containing the trainings data
        :return: gnll loss vector, loss of the physical model
        """
        y_calc_tmp = y_calc.clone()

        # undo norm
        if data.norm_tgts_bounds:
            y_calc_tmp, _ = data.undo_norm_tgts(predictions=y_calc_tmp, prediction_tgts=None, both_toggle=False)

        # fix return shape
        batch_size_tmp = self.batch_size
        self.batch_size = len(y_tgt)

        if batch_size_tmp != self.batch_size:
            self.ret_shape = ones([self.batch_size, 31], dtype=pytFl32, device=self.dev)

            # fill parameter for the physical model that are constant
            if self.param_missing:
                for i, index in enumerate(self.param_missing):
                    if index == 'dB0':
                        self.b0_shift = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.b0_shift[0]
                    elif index == 'B1':
                        self.b1 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.b1[0]
                    elif index == 'T1':
                        self.t1 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.t1[0]
                    elif index == 'T2':
                        self.t2 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.t2[0]
                    else:
                        raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                        'only exists for these values'.format(index))

        # fill parameter for the physical model
        for i, index in enumerate(self.param):
            if index == 'dB0':
                self.b0_shift = y_calc_tmp[:, i].to(self.dev)
            elif index == 'B1':
                self.b1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T1':
                self.t1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T2':
                self.t2 = y_calc_tmp[:, i].to(self.dev)
            else:
                raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                'only exists for these values'.format(index))

        # calc wasabiti spectra w. physical model
        wasabiti_ret = wasabiti_alt(self.offsets,
                                    self.trec,
                                    self.b0_shift,
                                    self.b1,
                                    self.t1,
                                    self.t2,
                                    self.freq,
                                    self.tp,
                                    self.gamma,
                                    self.ret_shape)

        # first term
        ret = mul(sum(square(div(sub(y_tgt,
                                     y_calc[:, :self.n]),  # subtract
                                 (y_calc[:, self.n:]))  # divide
                             ),  # square
                      dim=1),  # sum up the tensor
                  0.5)  # multiply the coefficient

        # second term
        ret += sum(log(abs(y_calc[:, self.n:])), dim=1)

        # third term
        ret += mul(self.third_term_const, self.n)

        # sum up and divide by M via mean()
        return ret*0., self.mse_mean(wasabiti_ret, input_x)




class GNLLonSigma_phys_m_new(nn.Module):
    """Calculates the gaussian log-likelihood loss + lambda * physical model MSE. Trains on uncertainties = sigma.
    This also considers "relaxation during dead_time (30 µs) and spoiler (5.5 ms)"

    It also takes the abs of sigma to avoid problems with the log(sigma). This is also done for the prediction
    """
    def __init__(self, n: Tensor,
                 dev: device,
                 param: list,
                 x: Union[Tensor, list],
                 trec: Union[Tensor, list, float],
                 lambda_fact: Tensor = 1.,
                 b0_shift: Tensor = 0.,
                 b1: Tensor = 3.75,
                 t1: Tensor = 2.,
                 t2: Tensor = 0.1,
                 tp: Tensor = 0.005,
                 gamma: Tensor = 42.5764):
        """

        :param n: number of tgt parameter
        :param dev: device
        :param param:
        :param lambda_fact:
        :param x: frequency offsets [ppm]
        :param trec: recover time between different offsets/measurements. Can be a float or an array [s]
        :param b0_shift: b0_shift-shift [ppm]
        :param b1: B1 peak amplitude [µT]
        :param t1: longitudinal relaxation time (T_1) [s]
        :param t2: transversal relaxation time (T_2) [s]
        :param tp: duration of the WASABI pulse [s]
        :param gamma: gyromagnetic ratio [MHz/T]
        """
        super().__init__()
        self.dev = dev
        self.third_term_const = 0.5 * log(tensor(2 * pi, device=dev))
        self.n = n  # number of parameters

        # physical model params
        self.offsets = tensor(x, dtype=pytFl32, device=dev)
        self.trec = tensor(trec, dtype=pytFl32, device=dev)
        self.b0_shift = tensor([b0_shift], dtype=pytFl32, device=dev)
        self.b1 = tensor([b1], dtype=pytFl32, device=dev)
        self.t1 = tensor([t1], dtype=pytFl32, device=dev)
        self.t2 = tensor([t2], dtype=pytFl32, device=dev)
        self.gamma = tensor(gamma, dtype=pytFl32, device=dev)
        self.freq = tensor(gamma * 3., dtype=pytFl32, device=dev)
        self.tp = tensor(tp, dtype=pytFl32, device=dev)

        # scaling factor of the physical model
        self.lambda_fact = tensor(lambda_fact, dtype=pytFl32, device=dev)
        # parameter list
        self.param = param
        self.param_missing = list({'dB0', 'B1', 'T1', 'T2'} - set(self.param))


        # batch size
        self.batch_size = 1
        # base return size, used to bring parameters to correct shape
        self.ret_shape = ones([31], dtype=pytFl32, device=dev)


        # initialize MSE for the physical model
        self.mse_mean = nn.MSELoss(reduction='mean')
        self.mse_sum = nn.MSELoss(reduction='sum')

    def forward(self,
                y_calc: Optional[Tensor],
                y_tgt: Optional[Tensor],
                input_x: Tensor,
                data: Data) -> [Tensor, Tensor]:
        """Calculates the gaussian log-likelihood loss.

        :param y_calc: mini-batch containing the features and uncertainties (in log(sigma) form)
        :param y_tgt: mini-batch containing the targets
        :param input_x: the z-spectra of that trainings batch
        :param data: the data object containing the trainings data
        :return: gnll loss vector, loss of the physical model
        """
        y_calc_tmp = y_calc.clone()

        # undo norm
        if data.norm_tgts_bounds:
            y_calc_tmp, _ = data.undo_norm_tgts(predictions=y_calc_tmp, prediction_tgts=None, both_toggle=False)

        # fix return shape
        batch_size_tmp = self.batch_size
        self.batch_size = len(y_tgt)

        if batch_size_tmp != self.batch_size:
            self.ret_shape = ones([self.batch_size, 31], dtype=pytFl32, device=self.dev)

            # fill parameter for the physical model that are constant
            if self.param_missing:
                for i, index in enumerate(self.param_missing):
                    if index == 'dB0':
                        self.b0_shift = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.b0_shift[0]
                    elif index == 'B1':
                        self.b1 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.b1[0]
                    elif index == 'T1':
                        self.t1 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.t1[0]
                    elif index == 'T2':
                        self.t2 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.t2[0]
                    else:
                        raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                        'only exists for these values'.format(index))

        # fill parameter for the physical model
        for i, index in enumerate(self.param):
            if index == 'dB0':
                self.b0_shift = y_calc_tmp[:, i].to(self.dev)
            elif index == 'B1':
                self.b1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T1':
                self.t1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T2':
                self.t2 = y_calc_tmp[:, i].to(self.dev)
            else:
                raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                'only exists for these values'.format(index))

        # calc wasabiti spectra w. physical model
        wasabiti_ret = wasabiti_new(self.offsets,
                                    self.trec,
                                    self.b0_shift,
                                    self.b1,
                                    self.t1,
                                    self.t2,
                                    self.tp,
                                    self.gamma,
                                    self.ret_shape)

        # first term
        ret = mul(sum(square(div(sub(y_tgt,
                                     y_calc[:, :self.n]),  # subtract
                                 (y_calc[:, self.n:]))  # divide
                             ),  # square
                      dim=1),  # sum up the tensor
                  0.5)  # multiply the coefficient

        # second term
        ret += sum(log(abs(y_calc[:, self.n:])), dim=1)

        # third term
        ret += mul(self.third_term_const, self.n)

        # sum up and divide by M via mean()
        return ret, self.mse_mean(wasabiti_ret, input_x)


class GNLLonSigma_phys_m_alt_noNorm(nn.Module):
    """Calculates the gaussian log-likelihood loss + lambda * pyhsical model (including relaxation during dead_time
    (30 µs) and spoiler (5.5 ms)) MSE. Trains on uncertainties = sigma. It also undoes the norm before calculating the
    GNLL.

    It also takes the abs of sigma to avoid problems with the log(sigma). This is also done for the prediction
    """
    def __init__(self, n: Tensor,
                 dev: device,
                 param: list,
                 x: Union[Tensor, list],
                 trec: Union[Tensor, list, float],
                 lambda_fact: Tensor = 1.,
                 b0_shift: Tensor = 0.,
                 b1: Tensor = 3.75,
                 t1: Tensor = 2.,
                 t2: Tensor = 0.1,
                 tp: Tensor = 0.005,
                 gamma: Tensor = 42.5764):
        """

        :param n: number of tgt parameter
        :param dev: device
        :param param:
        :param lambda_fact:
        :param x: frequency offsets [ppm]
        :param trec: recover time between different offsets/measurements. Can be a float or an array [s]
        :param b0_shift: b0_shift-shift [ppm]
        :param b1: B1 peak amplitude [µT]
        :param t1: longitudinal relaxation time (T_1) [s]
        :param t2: transversal relaxation time (T_2) [s]
        :param tp: duration of the WASABI pulse [s]
        :param gamma: gyromagnetic ratio [MHz/T]
        """
        super().__init__()
        self.dev = dev
        self.third_term_const = 0.5 * log(tensor(2 * pi, device=dev))
        self.n = n  # number of parameters

        # physical model params
        self.offsets = tensor(x, dtype=pytFl32, device=dev)
        self.trec = tensor(trec, dtype=pytFl32, device=dev)
        self.b0_shift = tensor([b0_shift], dtype=pytFl32, device=dev)
        self.b1 = tensor([b1], dtype=pytFl32, device=dev)
        self.t1 = tensor([t1], dtype=pytFl32, device=dev)
        self.t2 = tensor([t2], dtype=pytFl32, device=dev)
        self.gamma = tensor(gamma, dtype=pytFl32, device=dev)
        self.freq = tensor(gamma * 3., dtype=pytFl32, device=dev)
        self.tp = tensor(tp, dtype=pytFl32, device=dev)

        # scaling factor of the physical model
        self.lambda_fact = tensor(lambda_fact, dtype=pytFl32, device=dev)
        # parameter list
        self.param = param
        self.param_missing = list({'dB0', 'B1', 'T1', 'T2'} - set(self.param))


        # batch size
        self.batch_size = 1
        # base return size, used to bring parameters to correct shape
        self.ret_shape = ones([31], dtype=pytFl32, device=dev)


        # initialize MSE for the physical model
        self.mse_mean = nn.MSELoss(reduction='mean')
        self.mse_sum = nn.MSELoss(reduction='sum')

    def forward(self,
                y_calc: Optional[Tensor],
                y_tgt: Optional[Tensor],
                input_x: Tensor,
                data: Data) -> [Tensor, Tensor]:
        """Calculates the gaussian log-likelihood loss.

        :param y_calc: mini-batch containing the features and uncertainties (in log(sigma) form)
        :param y_tgt: mini-batch containing the targets
        :param input_x: the z-spectra of that trainings batch
        :param data: the data object containing the trainings data
        :return: gnll loss vector, loss of the physical model
        """
        y_calc_tmp = y_calc.clone()
        y_tgt_tmp = y_tgt.clone()

        # undo norm
        if data.norm_tgts_bounds:
            y_calc_tmp, y_tgt_tmp = data.undo_norm_tgts(predictions=y_calc_tmp, prediction_tgts=y_tgt_tmp, both_toggle=True)

        # fix return shape
        batch_size_tmp = self.batch_size
        self.batch_size = len(y_tgt)

        if batch_size_tmp != self.batch_size:
            self.ret_shape = ones([self.batch_size, 31], dtype=pytFl32, device=self.dev)

            # fill parameter for the physical model that are constant
            if self.param_missing:
                for i, index in enumerate(self.param_missing):
                    if index == 'dB0':
                        self.b0_shift = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.b0_shift[0]
                    elif index == 'B1':
                        self.b1 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.b1[0]
                    elif index == 'T1':
                        self.t1 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.t1[0]
                    elif index == 'T2':
                        self.t2 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.t2[0]
                    else:
                        raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                        'only exists for these values'.format(index))

        # fill parameter for the physical model
        for i, index in enumerate(self.param):
            if index == 'dB0':
                self.b0_shift = y_calc_tmp[:, i].to(self.dev)
            elif index == 'B1':
                self.b1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T1':
                self.t1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T2':
                self.t2 = y_calc_tmp[:, i].to(self.dev)
            else:
                raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                'only exists for these values'.format(index))

        # calc wasabiti spectra w. physical model
        wasabiti_ret = wasabiti_alt(self.offsets,
                                    self.trec,
                                    self.b0_shift,
                                    self.b1,
                                    self.t1,
                                    self.t2,
                                    self.freq,
                                    self.tp,
                                    self.gamma,
                                    self.ret_shape)

        # first term
        ret = mul(sum(square(div(sub(y_tgt_tmp,
                                     y_calc_tmp[:, :self.n]),  # subtract
                                 (y_calc_tmp[:, self.n:]))  # divide
                             ),  # square
                      dim=1),  # sum up the tensor
                  0.5)  # multiply the coefficient

        # second term
        ret += sum(log(abs(y_calc_tmp[:, self.n:])), dim=1)

        # third term
        ret += mul(self.third_term_const, self.n)

        # sum up and divide by M via mean()
        return ret, self.mse_mean(wasabiti_ret, input_x)


class GNLLonSigma_phys_m_alt_useTgts(nn.Module):
    """Calculates the gaussian log-likelihood loss + lambda * pyhsical model (including relaxation during dead_time
    (30 µs) and spoiler (5.5 ms)) MSE. Trains on uncertainties = sigma.
    This version uses the target parameter for all non-trained parameters in the PhysM.
    IMPORTANT: This requires the target parameters in the order {'dB0', 'B1', 'T1', 'T2'}!

    It also takes the abs of sigma to avoid problems with the log(sigma). This is also done for the prediction
    """
    def __init__(self, n: Tensor,
                 dev: device,
                 param: list,
                 x: Union[Tensor, list],
                 trec: Union[Tensor, list, float],
                 lambda_fact: Tensor = 1.,
                 b0_shift: Tensor = 0.,
                 b1: Tensor = 3.75,
                 t1: Tensor = 2.,
                 t2: Tensor = 0.1,
                 tp: Tensor = 0.005,
                 gamma: Tensor = 42.5764):
        """

        :param n: number of tgt parameter
        :param dev: device
        :param param:
        :param lambda_fact:
        :param x: frequency offsets [ppm]
        :param trec: recover time between different offsets/measurements. Can be a float or an array [s]
        :param b0_shift: b0_shift-shift [ppm]
        :param b1: B1 peak amplitude [µT]
        :param t1: longitudinal relaxation time (T_1) [s]
        :param t2: transversal relaxation time (T_2) [s]
        :param tp: duration of the WASABI pulse [s]
        :param gamma: gyromagnetic ratio [MHz/T]
        """
        super().__init__()
        self.dev = dev
        self.third_term_const = 0.5 * log(tensor(2 * pi, device=dev))
        self.n = n  # number of parameters

        # physical model params
        self.offsets = tensor(x, dtype=pytFl32, device=dev)
        self.trec = tensor(trec, dtype=pytFl32, device=dev)
        self.b0_shift = tensor([b0_shift], dtype=pytFl32, device=dev)
        self.b1 = tensor([b1], dtype=pytFl32, device=dev)
        self.t1 = tensor([t1], dtype=pytFl32, device=dev)
        self.t2 = tensor([t2], dtype=pytFl32, device=dev)
        self.gamma = tensor(gamma, dtype=pytFl32, device=dev)
        self.freq = tensor(gamma * 3., dtype=pytFl32, device=dev)
        self.tp = tensor(tp, dtype=pytFl32, device=dev)

        # scaling factor of the physical model
        self.lambda_fact = tensor(lambda_fact, dtype=pytFl32, device=dev)
        # parameter list
        self.param = param
        self.param_missing = list({'dB0', 'B1', 'T1', 'T2'} - set(self.param))


        # batch size
        self.batch_size = 1
        # base return size, used to bring parameters to correct shape
        self.ret_shape = ones([31], dtype=pytFl32, device=dev)


        # initialize MSE for the physical model
        self.mse_mean = nn.MSELoss(reduction='mean')
        self.mse_sum = nn.MSELoss(reduction='sum')

    def forward(self,
                y_calc: Optional[Tensor],
                y_tgt: Optional[Tensor],
                input_x: Tensor,
                y_no_noise_all: Tensor,
                data: Data) -> [Tensor, Tensor]:
        """Calculates the gaussian log-likelihood loss.

        :param y_calc: mini-batch containing the features and uncertainties (in log(sigma) form)
        :param y_tgt: mini-batch containing the targets
        :param input_x: the z-spectra of that trainings batch
        :param y_no_noise_all: all possible target parameter in order without noise, assumed to be not normed
        :param data: the data object containing the trainings data
        :return: gnll loss vector, loss of the physical model
        """
        y_calc_tmp = y_calc.clone()
        y_no_noise_all_tmp = y_no_noise_all.clone()

        # undo norm
        if data.norm_tgts_bounds:
            # trained params for PhysM
            y_calc_tmp, _ = data.undo_norm_tgts(predictions=y_calc_tmp,
                                                        prediction_tgts=None,
                                                        both_toggle=False)
            # # not trained params for PhysM
            # y_no_noise_all_tmp, _ = data.undo_norm_tgts(predictions=y_no_noise_all_tmp,
            #                                            prediction_tgts=None,
            #                                            both_toggle=False,
            #                                            all_toggle=True)


        # fix return shape
        batch_size_tmp = self.batch_size
        self.batch_size = len(y_tgt)

        if batch_size_tmp != self.batch_size:
            self.ret_shape = ones([self.batch_size, 31], dtype=pytFl32, device=self.dev)

            # fill the parameter for the physical model that are not trained with the true targets
            if self.param_missing:
                for i, index in enumerate(self.param_missing):
                    if index == 'dB0':
                        self.b0_shift = y_no_noise_all_tmp[:, 0].clone()
                    elif index == 'B1':
                        self.b1 = y_no_noise_all_tmp[:, 1].clone()
                    elif index == 'T1':
                        self.t1 = y_no_noise_all_tmp[:, 2].clone()
                    elif index == 'T2':
                        self.t2 = y_no_noise_all_tmp[:, 3].clone()
                    else:
                        raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                        'only exists for these values'.format(index))

        # fill parameter for the physical model
        for i, index in enumerate(self.param):
            if index == 'dB0':
                self.b0_shift = y_calc_tmp[:, i].to(self.dev)
            elif index == 'B1':
                self.b1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T1':
                self.t1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T2':
                self.t2 = y_calc_tmp[:, i].to(self.dev)
            else:
                raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                'only exists for these values'.format(index))

        # calc wasabiti spectra w. physical model
        wasabiti_ret = wasabiti_alt(self.offsets,
                                    self.trec,
                                    self.b0_shift,
                                    self.b1,
                                    self.t1,
                                    self.t2,
                                    self.freq,
                                    self.tp,
                                    self.gamma,
                                    self.ret_shape)

        # first term
        ret = mul(sum(square(div(sub(y_tgt,
                                     y_calc[:, :self.n]),  # subtract
                                 (y_calc[:, self.n:]))  # divide
                             ),  # square
                      dim=1),  # sum up the tensor
                  0.5)  # multiply the coefficient

        # second term
        ret += sum(log(abs(y_calc[:, self.n:])), dim=1)

        # third term
        ret += mul(self.third_term_const, self.n)

        # sum up and divide by M via mean()
        return ret, self.mse_mean(wasabiti_ret, input_x)


class GNLLonSigma_phys_m_alt(nn.Module):
    """Calculates the gaussian log-likelihood loss + lambda * pyhsical model (including relaxation during dead_time
    (30 µs) and spoiler (5.5 ms)) MSE. Trains on uncertainties = sigma.

    It also takes the abs of sigma to avoid problems with the log(sigma). This is also done for the prediction
    """
    def __init__(self, n: Tensor,
                 dev: device,
                 param: list,
                 x: Union[Tensor, list],
                 trec: Union[Tensor, list, float],
                 lambda_fact: Tensor = 1.,
                 b0_shift: Tensor = 0.,
                 b1: Tensor = 3.75,
                 t1: Tensor = 2.,
                 t2: Tensor = 0.1,
                 tp: Tensor = 0.005,
                 gamma: Tensor = 42.5764):
        """

        :param n: number of tgt parameter
        :param dev: device
        :param param:
        :param lambda_fact:
        :param x: frequency offsets [ppm]
        :param trec: recover time between different offsets/measurements. Can be a float or an array [s]
        :param b0_shift: b0_shift-shift [ppm]
        :param b1: B1 peak amplitude [µT]
        :param t1: longitudinal relaxation time (T_1) [s]
        :param t2: transversal relaxation time (T_2) [s]
        :param tp: duration of the WASABI pulse [s]
        :param gamma: gyromagnetic ratio [MHz/T]
        """
        super().__init__()
        self.dev = dev
        self.third_term_const = 0.5 * log(tensor(2 * pi, device=dev))
        self.n = n  # number of parameters

        # physical model params
        self.offsets = tensor(x, dtype=pytFl32, device=dev)
        self.trec = tensor(trec, dtype=pytFl32, device=dev)
        self.b0_shift = tensor([b0_shift], dtype=pytFl32, device=dev)
        self.b1 = tensor([b1], dtype=pytFl32, device=dev)
        self.t1 = tensor([t1], dtype=pytFl32, device=dev)
        self.t2 = tensor([t2], dtype=pytFl32, device=dev)
        self.gamma = tensor(gamma, dtype=pytFl32, device=dev)
        self.freq = tensor(gamma * 3., dtype=pytFl32, device=dev)
        self.tp = tensor(tp, dtype=pytFl32, device=dev)

        # scaling factor of the physical model
        self.lambda_fact = tensor(lambda_fact, dtype=pytFl32, device=dev)
        # parameter list
        self.param = param
        self.param_missing = list({'dB0', 'B1', 'T1', 'T2'} - set(self.param))


        # batch size
        self.batch_size = 1
        # base return size, used to bring parameters to correct shape
        self.ret_shape = ones([31], dtype=pytFl32, device=dev)


        # initialize MSE for the physical model
        self.mse_mean = nn.MSELoss(reduction='mean')
        self.mse_sum = nn.MSELoss(reduction='sum')

    def forward(self,
                y_calc: Optional[Tensor],
                y_tgt: Optional[Tensor],
                input_x: Tensor,
                data: Data) -> [Tensor, Tensor]:
        """Calculates the gaussian log-likelihood loss.

        :param y_calc: mini-batch containing the features and uncertainties (in log(sigma) form)
        :param y_tgt: mini-batch containing the targets
        :param input_x: the z-spectra of that trainings batch
        :param data: the data object containing the trainings data
        :return: gnll loss vector, loss of the physical model
        """
        y_calc_tmp = y_calc.clone()

        # undo norm
        if data.norm_tgts_bounds:
            y_calc_tmp, _ = data.undo_norm_tgts(predictions=y_calc_tmp, prediction_tgts=None, both_toggle=False)

        # fix return shape
        batch_size_tmp = self.batch_size
        self.batch_size = len(y_tgt)

        if batch_size_tmp != self.batch_size:
            self.ret_shape = ones([self.batch_size, 31], dtype=pytFl32, device=self.dev)

            # fill parameter for the physical model that are constant
            if self.param_missing:
                for i, index in enumerate(self.param_missing):
                    if index == 'dB0':
                        self.b0_shift = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.b0_shift[0]
                    elif index == 'B1':
                        self.b1 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.b1[0]
                    elif index == 'T1':
                        self.t1 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.t1[0]
                    elif index == 'T2':
                        self.t2 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.t2[0]
                    else:
                        raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                        'only exists for these values'.format(index))

        # fill parameter for the physical model
        for i, index in enumerate(self.param):
            if index == 'dB0':
                self.b0_shift = y_calc_tmp[:, i].to(self.dev)
            elif index == 'B1':
                self.b1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T1':
                self.t1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T2':
                self.t2 = y_calc_tmp[:, i].to(self.dev)
            else:
                raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                'only exists for these values'.format(index))

        # calc wasabiti spectra w. physical model
        wasabiti_ret = wasabiti_alt(self.offsets,
                                    self.trec,
                                    self.b0_shift,
                                    self.b1,
                                    self.t1,
                                    self.t2,
                                    self.freq,
                                    self.tp,
                                    self.gamma,
                                    self.ret_shape)

        # first term
        ret = mul(sum(square(div(sub(y_tgt,
                                     y_calc[:, :self.n]),  # subtract
                                 (y_calc[:, self.n:]))  # divide
                             ),  # square
                      dim=1),  # sum up the tensor
                  0.5)  # multiply the coefficient

        # second term
        ret += sum(log(abs(y_calc[:, self.n:])), dim=1)

        # third term
        ret += mul(self.third_term_const, self.n)

        # sum up and divide by M via mean()
        return ret, self.mse_mean(wasabiti_ret, input_x)


class GNLLonSigma_phys_m(nn.Module):
    """Calculates the gaussian log-likelihood loss + lambda * pyhsical model MSE. Trains on uncertainties = sigma.

    It also takes the abs of sigma to avoid problems with the log(sigma). This is also done for the prediction
    """
    def __init__(self, n: Tensor,
                 dev: device,
                 param: list,
                 x: Union[Tensor, list],
                 trec: Union[Tensor, list, float],
                 lambda_fact: Tensor = 1.,
                 b0_shift: Tensor = 0.,
                 b1: Tensor = 3.75,
                 t1: Tensor = 2.,
                 t2: Tensor = 0.1,
                 tp: Tensor = 0.005,
                 gamma: Tensor = 42.5764):
        """

        :param n: number of tgt parameter
        :param dev: device
        :param param:
        :param lambda_fact:
        :param x: frequency offsets [ppm]
        :param trec: recover time between different offsets/measurements. Can be a float or an array [s]
        :param b0_shift: b0_shift-shift [ppm]
        :param b1: B1 peak amplitude [µT]
        :param t1: longitudinal relaxation time (T_1) [s]
        :param t2: transversal relaxation time (T_2) [s]
        :param tp: duration of the WASABI pulse [s]
        :param gamma: gyromagnetic ratio [MHz/T]
        """
        super().__init__()
        self.dev = dev
        self.third_term_const = 0.5 * log(tensor(2 * pi, device=dev))
        self.n = n  # number of parameters

        # physical model params
        self.offsets = tensor(x, dtype=pytFl32, device=dev)
        self.trec = tensor(trec, dtype=pytFl32, device=dev)
        self.b0_shift = tensor([b0_shift], dtype=pytFl32, device=dev)
        self.b1 = tensor([b1], dtype=pytFl32, device=dev)
        self.t1 = tensor([t1], dtype=pytFl32, device=dev)
        self.t2 = tensor([t2], dtype=pytFl32, device=dev)
        self.gamma = tensor(gamma, dtype=pytFl32, device=dev)
        self.freq = tensor(gamma * 3., dtype=pytFl32, device=dev)
        self.tp = tensor(tp, dtype=pytFl32, device=dev)

        # scaling factor of the physical model
        self.lambda_fact = tensor(lambda_fact, dtype=pytFl32, device=dev)
        # parameter list
        self.param = param
        self.param_missing = list({'dB0', 'B1', 'T1', 'T2'} - set(self.param))


        # batch size
        self.batch_size = 1
        # base return size, used to bring parameters to correct shape
        self.ret_shape = ones([31], dtype=pytFl32, device=dev)


        # initialize MSE for the physical model
        self.mse_mean = nn.MSELoss(reduction='mean')
        self.mse_sum = nn.MSELoss(reduction='sum')

    def forward(self,
                y_calc: Optional[Tensor],
                y_tgt: Optional[Tensor],
                input_x: Tensor,
                data: Data) -> [Tensor, Tensor]:
        """Calculates the gaussian log-likelihood loss.

        :param y_calc: mini-batch containing the features and uncertainties (in log(sigma) form)
        :param y_tgt: mini-batch containing the targets
        :param input_x: the z-spectra of that trainings batch
        :param data: the data object containing the trainings data
        :return: gnll loss vector, loss of the physical model
        """
        y_calc_tmp = y_calc.clone()

        # undo norm
        if data.norm_tgts_bounds:
            y_calc_tmp, _ = data.undo_norm_tgts(predictions=y_calc_tmp, prediction_tgts=None, both_toggle=False)

        # fix return shape
        batch_size_tmp = self.batch_size
        self.batch_size = len(y_tgt)

        if batch_size_tmp != self.batch_size:
            self.ret_shape = ones([self.batch_size, 31], dtype=pytFl32, device=self.dev)

            # fill parameter for the physical model that are constant
            if self.param_missing:
                for i, index in enumerate(self.param_missing):
                    if index == 'dB0':
                        self.b0_shift = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.b0_shift[0]
                    elif index == 'B1':
                        self.b1 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.b1[0]
                    elif index == 'T1':
                        self.t1 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.t1[0]
                    elif index == 'T2':
                        self.t2 = ones_like(y_calc_tmp[:, i], dtype=pytFl32, device=self.dev) * self.t2[0]
                    else:
                        raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                        'only exists for these values'.format(index))

        # fill parameter for the physical model
        for i, index in enumerate(self.param):
            if index == 'dB0':
                self.b0_shift = y_calc_tmp[:, i].to(self.dev)
            elif index == 'B1':
                self.b1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T1':
                self.t1 = y_calc_tmp[:, i].to(self.dev)
            elif index == 'T2':
                self.t2 = y_calc_tmp[:, i].to(self.dev)
            else:
                raise NameError('{} is not dB0, B1, T1 or T2. This loss function '
                                'only exists for these values'.format(index))

        # calc wasabiti spectra w. physical model
        wasabiti_ret = wasabiti(self.offsets,
                                self.trec,
                                self.b0_shift,
                                self.b1,
                                self.t1,
                                self.t2,
                                self.freq,
                                self.tp,
                                self.gamma,
                                self.ret_shape)

        # first term
        ret = mul(sum(square(div(sub(y_tgt,
                                     y_calc[:, :self.n]),  # subtract
                                 (y_calc[:, self.n:]))  # divide
                             ),  # square
                      dim=1),  # sum up the tensor
                  0.5)  # multiply the coefficient

        # second term
        ret += sum(log(abs(y_calc[:, self.n:])), dim=1)

        # third term
        ret += mul(self.third_term_const, self.n)

        # sum up and divide by M via mean()
        return ret, self.mse_mean(wasabiti_ret, input_x)


class GNLL(nn.Module):
    """Calculates the gaussian log-likelihood loss. Trains on uncertainties = ln(sigma)."""
    def __init__(self, n: Tensor, dev: device):
        super().__init__()
        self.dev = dev
        self.third_term_const = 0.5 * log(tensor(2 * pi, device=dev))
        self.n = n  # number of parameters

    def forward(self,
                y_calc: Optional[Tensor],
                y_tgt: Optional[Tensor]) -> [Tensor, Tensor]:
        """Calculates the gaussian log-likelihood loss.

        :param y_calc: mini-batch containing the features and uncertainties (in log(sigma) form)
        :param y_tgt: mini-batch containing the targets
        :return: mean loss, summed up loss
        """
        # first term
        ret = mul(sum(square(div(sub(y_tgt,
                                     y_calc[:, :self.n]),  # subtract
                                 exp(y_calc[:, self.n:]))  # divide
                             ),  # square
                      dim=1),  # sum up the tensor
                  0.5)  # multiply the coefficient

        # second term
        ret += sum(y_calc[:, self.n:], dim=1)

        # third term
        ret += mul(self.third_term_const, self.n)

        # sum up and divide by M via mean()
        return ret.mean(), ret.sum()


class GNLLonSigma(nn.Module):
    """Calculates the gaussian log-likelihood loss. Trains on uncertainties = sigma.

    It also takes the abs of sigma to avoid problems with the log(sigma). This is also done for the prediction
    """
    def __init__(self, n: Tensor, dev: device):
        super().__init__()
        self.dev = dev
        self.third_term_const = 0.5 * log(tensor(2 * pi, device=dev))
        self.n = n  # number of parameters

    def forward(self,
                y_calc: Optional[Tensor],
                y_tgt: Optional[Tensor]) -> [Tensor, Tensor]:
        """Calculates the gaussian log-likelihood loss.

        :param y_calc: mini-batch containing the features and uncertainties (in log(sigma) form)
        :param y_tgt: mini-batch containing the targets
        :return: mean loss, summed up loss
        """
        # first term
        ret = mul(sum(square(div(sub(y_tgt,
                                     y_calc[:, :self.n]),  # subtract
                                 (y_calc[:, self.n:]))  # divide
                             ),  # square
                      dim=1),  # sum up the tensor
                  0.5)  # multiply the coefficient

        # second term
        ret += sum(log(abs(y_calc[:, self.n:])), dim=1)

        # third term
        ret += mul(self.third_term_const, self.n)

        # sum up and divide by M via mean()
        return ret.mean(), ret.sum()


class alt_MSELoss(nn.Module):
    """Calculates the MSELoss normally, except that it returns both the batch mean and the batch sum of the error."""
    def __init__(self, **kwargs):
        """

        :param kwargs: the kwargs for nn.MSELoss()
        """
        super().__init__()
        self.calc_mean = nn.MSELoss(**kwargs)
        self.calc_sum = nn.MSELoss(reduction='sum')

    def forward(self,
                x: Tensor,
                y: Tensor) -> [Tensor, Tensor]:
        """Calculates the MSELoss normally, except that it returns both the batch mean and the batch sum of the error.

        :param x: mini-batch containing the features
        :param y: mini-batch containing the targets
        :return: mean loss, summed up loss
        """
        ret = self.calc_mean(x, y)
        ret2 = self.calc_sum(x, y)
        return ret, ret2
