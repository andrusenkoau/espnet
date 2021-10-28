# ! /usr/bin/python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType
from nemo.utils import logging


class SDLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
            "target_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, reduction='mean_batch', backend='k2', loss_type='ctc', loss_batch_size=0, **loss_kwargs):
        super().__init__()
        self._blank = num_classes
        self.loss_batch_size = loss_batch_size
        if reduction == 'mean_batch':
            ctc_reduction = 'none'
            self._apply_batch_mean = True
        elif reduction in ['sum', 'mean', 'none']:
            ctc_reduction = reduction
            self._apply_batch_mean = False

        # we assume that self._blank + 1 == num_classes
        if backend == 'k2':
            if loss_type == 'ctc':
                from nemo.collections.asr.parts.k2.ctc import CTCLoss as K2Loss
            elif loss_type == 'sd':
                from nemo.collections.asr.parts.k2.sd import SDLoss as K2Loss

            self._loss = K2Loss(num_classes=self._blank+1, blank=self._blank, reduction=ctc_reduction, **loss_kwargs)
        elif backend == 'gtn':
            from nemo.collections.asr.parts.gtn.ctc import CTCLoss as GTNCTCLoss

            self._loss = GTNCTCLoss(blank_idx=self._blank, reduction=ctc_reduction)

    @typecheck()
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # override forward implementation
        # custom logic, if necessary

        assert not (torch.isnan(log_probs).any() or torch.isinf(log_probs).any())

        log_probs = log_probs.float()
        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()
        targets = targets.long()
        batch_size = log_probs.shape[0]
        if self.loss_batch_size > 0 and self.loss_batch_size < batch_size:
            loss_list = []
            for batch_idx in range(0, batch_size, self.loss_batch_size):
                begin = batch_idx
                end = min(begin + self.loss_batch_size, batch_size)
                loss_part = self._loss(
                    log_probs=log_probs[begin:end],
                    targets=targets[begin:end],
                    input_lengths=input_lengths[begin:end],
                    target_lengths=target_lengths[begin:end],
                )
                loss_list.append(loss_part)
            loss = torch.cat(loss_list, 0)
        else:
            loss = self._loss(
                log_probs=log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )
        if self._apply_batch_mean:
            # torch.mean gives nan if loss is empty
            loss = torch.mean(loss) if loss.nelement() > 0 else torch.sum(loss)
        return loss
