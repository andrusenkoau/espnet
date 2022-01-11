import pytest
import torch

from espnet2.asr.ctc import CTC


@pytest.fixture
def ctc_args():
    bs = 2
    h = torch.randn(bs, 10, 10)
    h_lens = torch.LongTensor([10, 8])
    y = torch.randint(0, 4, [2, 5])
    y_lens = torch.LongTensor([5, 2])
    return h, h_lens, y, y_lens



@pytest.mark.parametrize("ctc_type", ["builtin", "warpctc", "ctc-crf"])
def test_ctc_forward_backward(ctc_type, ctc_args):
    if ctc_type == "warpctc":
        pytest.importorskip("warpctc_pytorch")
    if ctc_type == "ctc-crf":
        den_lm_path = "/home/laptev/Projects/espnet/espnet-gnroy/assets/den_lm.fst"
        token_lm_path = "/home/laptev/Projects/espnet/espnet-gnroy/assets/phone_lm.fst"
        ctc = CTC(
            encoder_output_size=10,
            odim=5,
            ctc_type=ctc_type,
            den_lm_path=den_lm_path,
            token_lm_path=token_lm_path,
        )
    else:
        ctc = CTC(
            encoder_output_size=10,
            odim=5,
            ctc_type=ctc_type,
            den_lm_path=None,
            token_lm_path=None,
        )
    ctc(*ctc_args).sum().backward()


@pytest.mark.parametrize("ctc_type", ["builtin", "warpctc", "gtnctc"])
def test_ctc_log_softmax(ctc_type, ctc_args):
    if ctc_type == "warpctc":
        pytest.importorskip("warpctc_pytorch")
    ctc = CTC(
        encoder_output_size=10,
        odim=5,
        ctc_type=ctc_type,
        den_lm_path=None,
        token_lm_path=None,
    )
    ctc.log_softmax(ctc_args[0])


@pytest.mark.parametrize("ctc_type", ["builtin", "warpctc", "gtnctc"])
def test_ctc_argmax(ctc_type, ctc_args):
    if ctc_type == "warpctc":
        pytest.importorskip("warpctc_pytorch")
    ctc = CTC(
        encoder_output_size=10,
        odim=5,
        ctc_type=ctc_type,
        den_lm_path=None,
        token_lm_path=None,
    )
    ctc.argmax(ctc_args[0])
