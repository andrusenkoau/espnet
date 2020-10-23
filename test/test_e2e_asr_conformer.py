import argparse
import pytest
import torch

from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer import plot


def make_arg(**kwargs):
    defaults = dict(
        adim=2,
        aheads=1,
        dropout_rate=0.0,
        transformer_attn_dropout_rate=None,
        elayers=1,
        eunits=2,
        dlayers=1,
        dunits=2,
        sym_space="<space>",
        sym_blank="<blank>",
        transformer_decoder_selfattn_layer_type="selfattn",
        transformer_encoder_pos_enc_layer_type="rel_pos",
        transformer_encoder_selfattn_layer_type="rel_selfattn",
        macaron_style=True,
        use_cnn_module=True,
        cnn_module_kernel=3,
        transformer_init="pytorch",
        transformer_input_layer="conv2d",
        transformer_length_normalized_loss=True,
        report_cer=False,
        report_wer=False,
        mtlalpha=0.0,
        lsm_weight=0.001,
        char_list=["<blank>", "a", "e", "i", "o", "u"],
        ctc_type="warpctc",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def prepare(args):
    idim = 10
    odim = 3
    batchsize = 2
    ilens = [10, 9]
    olens = [3, 4]
    n_token = odim - 1
    model = E2E(idim, odim, args)
    x = torch.randn(batchsize, max(ilens), idim)
    y = (torch.rand(batchsize, max(olens)) * n_token % n_token).long()
    for i in range(batchsize):
        x[i, ilens[i] :] = -1
        y[i, olens[i] :] = model.ignore_id

    data = []
    for i in range(batchsize):
        data.append(
            (
                "utt%d" % i,
                {
                    "input": [{"shape": [ilens[i], idim]}],
                    "output": [{"shape": [olens[i]]}],
                },
            )
        )
    return model, x, torch.tensor(ilens), y, data


conformer_mcnn_args = dict(
    transformer_encoder_pos_enc_layer_type="rel_pos",
    transformer_encoder_selfattn_layer_type="rel_selfattn",
    macaron_style=True,
    use_cnn_module=False,
)

conformer_mcnn_mmacaron_args = dict(
    transformer_encoder_pos_enc_layer_type="rel_pos",
    transformer_encoder_selfattn_layer_type="rel_selfattn",
    macaron_style=False,
    use_cnn_module=False,
)

conformer_mcnn_mmacaron_mrelattn_args = dict(
    transformer_encoder_pos_enc_layer_type="abs_pos",
    transformer_encoder_selfattn_layer_type="selfattn",
    macaron_style=False,
    use_cnn_module=False,
)


def test_traceable_and_scriptable():
    args = make_arg()
    model, x_pad, ilens, y, data = prepare(args)
    ys_in_pad, ys_out_pad = add_sos_eos(y, model.sos, model.eos, model.ignore_id)
    ys_mask = target_mask(ys_in_pad, model.ignore_id)
    src_mask = make_non_pad_mask(ilens).unsqueeze(-2)
    model.eval()

    with torch.no_grad():
        hs_pad, hs_mask = model.encoder(x_pad, src_mask)
        pred_pad, pred_mask = model.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        # test encoder traceable
        traced_encoder = torch.jit.trace(model.encoder, (x_pad, src_mask))

        # test results equal
        hs_pad_tr, hs_mask_tr = traced_encoder(x_pad, src_mask)
        assert torch.all(hs_pad_tr.eq(hs_pad))
        assert torch.all(hs_mask_tr.eq(hs_mask))

        # test encoder scriptable
        model.encoder.scripting_prep()
        scripted_encoder = torch.jit.script(model.encoder)

        # test results equal
        hs_pad_scr, hs_mask_scr = scripted_encoder(x_pad, src_mask)
        assert torch.all(hs_pad_scr.eq(hs_pad))
        assert torch.all(hs_mask_scr.eq(hs_mask))

        # test decoder traceable
        traced_decoder = torch.jit.trace(
            model.decoder, (ys_in_pad, ys_mask, hs_pad, hs_mask)
        )

        # test results equal
        pred_pad_tr, pred_mask_tr = traced_decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        assert torch.all(pred_pad_tr.eq(pred_pad))
        assert torch.all(pred_mask_tr.eq(pred_mask))

        # test decoder scriptable
        model.decoder.scripting_prep()
        scripted_decoder = torch.jit.script(model.decoder)

        # test results equal
        pred_pad_scr, pred_mask_scr = scripted_decoder(
            ys_in_pad, ys_mask, hs_pad, hs_mask
        )
        assert torch.all(pred_pad_scr.eq(pred_pad))
        assert torch.all(pred_mask_scr.eq(pred_mask))


def _savefn(*args, **kwargs):
    return


@pytest.mark.parametrize(
    "model_dict",
    [
        {},
        conformer_mcnn_args,
        conformer_mcnn_mmacaron_args,
        conformer_mcnn_mmacaron_mrelattn_args,
    ],
)
def test_transformer_trainable_and_decodable(model_dict):
    args = make_arg(**model_dict)
    model, x, ilens, y, data = prepare(args)

    # check for pure CTC and pure Attention
    if args.mtlalpha == 1:
        assert model.decoder is None
    elif args.mtlalpha == 0:
        assert model.ctc is None

    # test beam search
    recog_args = argparse.Namespace(
        beam_size=1,
        penalty=0.0,
        ctc_weight=0.0,
        maxlenratio=1.0,
        lm_weight=0,
        minlenratio=0,
        nbest=1,
    )
    # test trainable
    optim = torch.optim.Adam(model.parameters(), 0.01)
    loss = model(x, ilens, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    # test attention plot
    attn_dict = model.calculate_all_attentions(x[0:1], ilens[0:1], y[0:1])
    plot.plot_multi_head_attention(data, attn_dict, "", savefn=_savefn)

    # test CTC plot
    ctc_probs = model.calculate_all_ctc_probs(x[0:1], ilens[0:1], y[0:1])
    if args.mtlalpha > 0:
        print(ctc_probs.shape)
    else:
        assert ctc_probs is None

    # test decodable
    with torch.no_grad():
        nbest = model.recognize(x[0, : ilens[0]].numpy(), recog_args)
        print(y[0])
        print(nbest[0]["yseq"][1:-1])
