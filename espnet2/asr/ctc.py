import logging
import os
import subprocess
import tempfile
from typing import Optional

import torch
import torch.nn.functional as F
from typeguard import check_argument_types


class CTC(torch.nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin or warpctc
        reduce: reduce the CTC loss into a scalar
    """

    _ctc_types = ("builtin", "warpctc", "ctc-crf")

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        den_lm_path: Optional[str],
        token_lm_path: Optional[str],
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = True,
        lamb: float = 0.1,
        ctc_crf_eager_mode: bool = False,
        focal_gamma: float = 0.0,
        entropy_beta: float = 0.0,
    ):
        assert check_argument_types()
        assert (
            ctc_type in self._ctc_types
        ), f"ctc_type must be one of the following: {self._ctc_types}; {self.ctc_type}"
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        self.ignore_nan_grad = ignore_nan_grad
        self.lamb = lamb
        self.den_lm_path = den_lm_path
        self.token_lm_path = token_lm_path
        self.ctc_crf_eager_mode = ctc_crf_eager_mode
        self.lms_are_set = False
        self.focal_gamma = focal_gamma
        self.entropy_beta = entropy_beta

        if self.ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(reduction="none")
        elif self.ctc_type == "warpctc":
            import warpctc_pytorch as warp_ctc

            if ignore_nan_grad:
                logging.warning("ignore_nan_grad option is not supported for warp_ctc")
            self.ctc_loss = warp_ctc.CTCLoss(size_average=False, reduce=False)
        elif self.ctc_type == "ctc-crf":
            from espnet2.asr.ctc_crf import CTC_CRF_LOSS

            if self.den_lm_path is None or not os.path.isfile(self.den_lm_path):
                logging.warning("den_lm_path for ctc-crf is not set or not valid.")

            if not self.ctc_crf_eager_mode and (
                self.token_lm_path is None or not os.path.isfile(self.token_lm_path)
            ):
                logging.warning("token_lm_path for ctc-crf is not set or not valid.")

            if not torch.cuda.is_available():
                logging.warning(
                    "CUDA is not available."
                    " Attempt to calculate the loss will result in segmentation fault."
                )

            if self.ctc_crf_eager_mode:
                logging.warning(
                    "ctc-crf eager mode is active. Constant path weights are not used."
                    " Expect negative and less representative loss values."
                )

            self.ctc_loss = CTC_CRF_LOSS(size_average=False, reduce=False, lamb=lamb, ignore_nan_grad=ignore_nan_grad)
        else:
            raise NotImplementedError

        self.reduce = reduce

    def __del__(self):
        if self.ctc_type == "ctc-crf" and self.lms_are_set:
            import ctc_crf_base

            # It is assumed to be running on a single visible GPU
            gpus = torch.IntTensor([0])
            ctc_crf_base.release_env(gpus)
            logging.info("den_lm released")

    def _compute_path_weight(self, th_target, th_olen) -> torch.Tensor:
        with tempfile.NamedTemporaryFile() as tmp:
            start = 0
            for i in range(len(th_olen)):
                target_str = " ".join(
                    [str(n) for n in (th_target[start : start + th_olen[i]]).tolist()]
                )
                tmp.write((f"name{i} {target_str}\n").encode())
                start += th_olen[i]
            tmp.flush()
            output = subprocess.check_output(
                ("path_weight", tmp.name, self.token_lm_path), stderr=subprocess.PIPE
            )
            path_weight = torch.Tensor(
                [
                    float(o.split()[-1])
                    for o in output.decode("utf-8").strip().split("\n")
                ]
            )
            return path_weight.mean()

    def _compute_focal_loss(self, loss, ilen):
        return (1 - torch.exp(-loss.div(ilen))) ** self.focal_gamma * loss

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        if self.ctc_type == "builtin":
            th_pred = th_pred.log_softmax(2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            if loss.requires_grad and self.ignore_nan_grad:
                # ctc_grad: (L, B, O)
                ctc_grad = loss.grad_fn(torch.ones_like(loss))
                ctc_grad = ctc_grad.sum([0, 2])
                indices = torch.isfinite(ctc_grad)
                size = indices.long().sum()
                if size == 0:
                    # Return as is
                    logging.warning(
                        "All samples in this mini-batch got nan grad."
                        " Returning nan value instead of CTC loss"
                    )
                elif size != th_pred.size(1):
                    logging.warning(
                        f"{th_pred.size(1) - size}/{th_pred.size(1)}"
                        " samples got nan grad."
                        " These were ignored for CTC loss."
                    )

                    # Create mask for target
                    target_mask = torch.full(
                        [th_target.size(0)],
                        1,
                        dtype=torch.bool,
                        device=th_target.device,
                    )
                    s = 0
                    for ind, le in enumerate(th_olen):
                        if not indices[ind]:
                            target_mask[s : s + le] = 0
                        s += le

                    # Calc loss again using maksed data
                    loss = self.ctc_loss(
                        th_pred[:, indices, :],
                        th_target[target_mask],
                        th_ilen[indices],
                        th_olen[indices],
                    )
            else:
                size = th_pred.size(1)
        elif self.ctc_type == "warpctc":
            # warpctc only supports float32
            th_pred = th_pred.to(dtype=torch.float32)
            size = th_pred.size(1)

            th_target = th_target.cpu().int()
            th_ilen = th_ilen.cpu().int()
            th_olen = th_olen.cpu().int()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
        elif self.ctc_type == "ctc-crf":
            # runtime den_lm initialization check
            if not self.lms_are_set:
                if self.den_lm_path is None or not os.path.isfile(self.den_lm_path):
                    raise ValueError(f'"den_lm_path" must be valid: {self.den_lm_path}')
                elif not self.ctc_crf_eager_mode and (
                    self.token_lm_path is None or not os.path.isfile(self.token_lm_path)
                ):
                    raise ValueError(
                        f'"token_lm_path" must be valid: {self.token_lm_path}'
                    )
                elif self.ctc_lo.weight.device.type != "cuda":
                    raise RuntimeError(
                        "ctc-crf must use GPU device to compute the loss."
                    )
                else:
                    import ctc_crf_base

                    # It is assumed to be running on a single visible GPU
                    # gpus = torch.IntTensor([0])
                    # print(f'[DEBUG]: th_pred.device is: {th_pred.device}')
                    gpus = torch.IntTensor([th_pred.device.index])
                    ctc_crf_base.init_env(self.den_lm_path, gpus)
                    logging.info("den_lm initialized")
                    self.lms_are_set = True

            th_pred = th_pred.log_softmax(2)
            size = th_pred.size(1)

            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            if not self.ctc_crf_eager_mode:
                loss_const = self._compute_path_weight(th_target, th_olen)
                loss -= loss_const
        else:
            raise NotImplementedError

        # Perform the maximum entropy regularization
        # Check more detils in Liu et al, 2018, "Connectionist Temporal Classification
        #                               with Maximum Entropy Regularization"
        # Note that this option is simpler and more aggressive
        # than proposed by Liu et al
        if self.entropy_beta != 0.0:
            if self.ctc_type == "warpctc":
                th_pred = th_pred.log_softmax(2)
            # Restore source loss values but keep gradients
            loss_source = loss.detach()
            loss = (1 - self.entropy_beta) * loss + self.entropy_beta * (
                (th_pred.exp() * th_pred).sum(2).sum(0)
            )
            loss += loss_source - loss.detach()

        # Compute focal CTC loss
        # Check more detils in Feng et al, 2019, "Focal CTC Loss for Chinese Optical
        #                               Character Recognition on Unbalanced Datasets"
        if self.focal_gamma != 0.0:
            # Restore source loss values but keep gradients
            loss_source = loss.detach()
            loss = self._compute_focal_loss(loss, th_ilen)
            loss += loss_source - loss.detach()

        if self.reduce:
            # Batch-size average
            loss = loss.sum() / size
        else:
            loss = loss / size

        return loss

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)

        # (B, L) -> (BxL,)
        ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )

        return loss

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
