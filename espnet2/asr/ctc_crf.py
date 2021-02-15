"""
Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang. Apache 2.0.
          2020-2021 ITMO University, Author: Aleksandr Laptev. Apache 2.0.

This script shows the implementation of CRF loss function.
Based on https://github.com/thu-spmi/CAT/
"""

import torch

import ctc_crf_base


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, "shouldn't require grads"


class _CRF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, input_lengths, label_lengths, size_average=True):
        logits = logits.transpose(0, 1).to(dtype=torch.float32)
        # For an unclear reason, gpu_den returns -inf for every second utterance.
        # So there is a workaround as follows.
        batch_size, seq_len, feat_dim = logits.size()
        logits_ext = torch.zeros(batch_size * 2, seq_len, feat_dim)
        logits_ext[::2] = logits.cpu()
        logits_ext = logits_ext.contiguous().to(logits.get_device())

        costs_alpha_den = torch.zeros(logits_ext.size(0)).type_as(logits_ext)
        costs_beta_den = torch.zeros(logits_ext.size(0)).type_as(logits_ext)

        grad = torch.zeros(logits_ext.size()).type_as(logits_ext)
        ctc_crf_base.gpu_den(
            logits_ext, grad, input_lengths.cuda(), costs_alpha_den, costs_beta_den
        )

        grad = grad[::2]
        costs = costs_alpha_den[::2]

        if size_average:
            grad = grad / batch_size
            costs = costs / batch_size

        ctx.grads = grad.transpose(0, 1)

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        return (
            ctx.grads.transpose(1, -1)
            .mul(grad_output.to(ctx.grads.device))
            .transpose(1, -1),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CTC_CRF_LOSS(torch.nn.Module):
    """CTC-CRF function module.

    :param float lamb: ctc smoothing coefficient (0.0 ~ 1.0)
    :param bool size_average: perform size average
    """

    def __init__(self, lamb=0.1, size_average=True, reduce=True):
        """Construct a CTC_CRF_LOSS object."""
        super(CTC_CRF_LOSS, self).__init__()
        self.crf = _CRF.apply
        self.ctc = torch.nn.CTCLoss(reduction="none")
        self.lamb = lamb
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, logits, labels, input_lengths, label_lengths):
        """CTC-CRF forward.

        :param torch.Tensor logits: batch of padded log prob sequences (Tmax, B, odim)
        :param torch.Tensor labels:
            batch of padded character id sequence tensor (B, Lmax)
        :param torch.Tensor input_lengths: batch of lengths of log prob sequences (B)
        :param torch.Tensor label_lengths: batch of lengths character id sequences (B)
        :return: ctc-crf loss value
        :rtype: torch.Tensor
        """
        assert len(labels.size()) == 1
        _assert_no_grad(labels)
        _assert_no_grad(input_lengths)
        _assert_no_grad(label_lengths)

        # crf only supports float32
        crf_cost = self.crf(
            logits,
            labels,
            input_lengths,
            label_lengths,
            self.size_average,
        )
        ctc_cost = self.ctc(logits, labels, input_lengths, label_lengths)
        if self.size_average:
            ctc_cost /= logits.size(1)
        cost = crf_cost + (1 + self.lamb) * ctc_cost
        if self.reduce:
            cost = cost.sum()
        return cost