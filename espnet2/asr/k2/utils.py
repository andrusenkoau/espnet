# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) 2020, Xiaomi CORPORATION.  All rights reserved.
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

import itertools
import os
import re
from pickle import UnpicklingError
from typing import List, Optional, Tuple, Union

import k2
import torch
import logging
#from nemo.utils import logging


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_with_grad: torch.Tensor, scale: float):
        ctx.save_for_backward(torch.tensor(scale))
        return tensor_with_grad

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.saved_tensors[0], None


class GradExpNormalize(torch.autograd.Function):
    def make_non_pad_mask(input_lengths: torch.Tensor, seq_len: int):
        batch_size = input_lengths.shape[0]
        seq_range = torch.arange(0, seq_len, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, seq_len)
        seq_length_expand = seq_range_expand.new(input_lengths.cpu()).unsqueeze(-1)
        mask = seq_range_expand < seq_length_expand
        return mask

    @staticmethod
    def forward(ctx, log_probs: torch.Tensor, input_lengths: torch.Tensor, reduction: str = "mean"):
        mask = GradExpNormalize.make_non_pad_mask(input_lengths, log_probs.shape[1])
        max_log_prob, _ = log_probs.max(-1)
        probs = torch.exp(log_probs - max_log_prob.unsqueeze(-1))
        norm_probs = torch.zeros_like(log_probs)
        norm_probs[mask] += (probs / probs.sum(-1).unsqueeze(-1))[mask]
        if reduction == "mean":
            norm_probs /= norm_probs.shape[0]
        ctx.save_for_backward(norm_probs)
        return log_probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output + ctx.saved_tensors[0], None, None


def compose_L_G(L: k2.Fsa, G: k2.Fsa):
    assert L.device == G.device
    LG = k2.connect(k2.compose(k2.arc_sort(L), k2.arc_sort(G)))
    LG = k2.connect(k2.determinize(LG))
    return LG.to(L.device)


def compose_T_LG(T: k2.Fsa, LG: k2.Fsa, labels_disambig_num: int = 2, aux_labels_disambig_num: int = 0):
    LG = LG.clone()
    T = T.clone()
    LG.convert_attr_to_ragged_(name='aux_labels')
    labels_disambig_id_start = LG.labels.max() - labels_disambig_num
    LG.labels[LG.labels > labels_disambig_id_start] = 0
    # LG.labels[LG.labels > 0] -= 1
    T.aux_labels[T.aux_labels > 0] += 1
    aux_labels_disambig_id_start = LG.aux_labels.values().max() - aux_labels_disambig_num
    LG.aux_labels.values()[LG.aux_labels.values() > aux_labels_disambig_id_start] = 0
    LG = k2.connect(k2.remove_epsilon(LG))
    LG.aux_labels = k2.ragged.remove_values_eq(LG.aux_labels, 0)
    TLG = k2.compose(T, k2.arc_sort(LG), inner_labels='phones')
    TLG = k2.arc_sort(k2.connect(TLG))
    # make blank_id = 0
    #if isinstance(TLG.phones, torch.Tensor):
    #    TLG.phones[TLG.phones > 0] -= 1
    #else:
    #    TLG.phones.values()[TLG.phones.values() > 0] -= 1
    return TLG


def create_supervision(input_lengths: torch.Tensor):
    supervisions = torch.stack(
        (
            torch.tensor(range(input_lengths.shape[0])),
            torch.zeros(input_lengths.shape[0]),
            input_lengths.cpu(),
        ),
        1,
    ).to(torch.int32)
    # the duration column has to be sorted in decreasing order
    order = torch.argsort(supervisions[:, -1], descending=True)
    return supervisions[order], order

def invert_permutation(indices: torch.Tensor) -> torch.Tensor:
    ans = torch.zeros(indices.shape, device=indices.device, dtype=torch.long)
    ans[indices] = torch.arange(0, indices.shape[0], device=indices.device)
    return ans

def make_blank_first(blank_idx: int, log_probs: torch.Tensor, targets: torch.Tensor):
    index = list(range(log_probs.shape[-1]))
    del index[blank_idx]
    index = torch.tensor([blank_idx] + index).to(log_probs.device)
    # TODO: fix this to work in general
    return torch.index_select(log_probs, -1, index), None if targets is None else targets + 1

def load_graph(graph_path):
    if os.path.exists(graph_path):
        errors = []
        try:
            graph_dict = torch.load(graph_path, map_location="cpu")
            graph = k2.Fsa.from_dict(graph_dict)
            return graph
        except UnpicklingError as e:
            errors.append(e)
            with open(graph_path, "rt", encoding="utf-8") as f:
                graph_txt = f.read()
            for func, acceptor in itertools.product([k2.Fsa.from_str, k2.Fsa.from_openfst], [True, False]):
                try:
                    graph = func(graph_txt, acceptor=acceptor)
                    logging("Success fsa reading")
                    return graph
                except (TypeError, ValueError, RuntimeError) as e:
                    errors.append(e)
        raise Exception(errors)
    else:
        logging.warning(f"""No such file: '{graph_path}'""")
        return None

def graph_to_den(graph: k2.Fsa, replicate_den: bool = False, times: int = 1) -> k2.Fsa:
    graph_vec = k2.create_fsa_vec([graph.detach()])
    if replicate_den:
        indexes = torch.zeros(times, dtype=torch.int32, device=graph.device)
        den = k2.index_fsa(graph_vec, indexes)
    else:
        den = graph_vec.to(graph.device)
    return den

def intersect_with_self_loops(base_graph: k2.Fsa, aux_graph: k2.Fsa) -> k2.Fsa:
    assert hasattr(base_graph, 'aux_labels')
    assert not hasattr(aux_graph, 'aux_labels')
    aux_graph_with_self_loops = k2.arc_sort(k2.add_epsilon_self_loops(aux_graph)).to(base_graph.device)
    result = k2.intersect(k2.arc_sort(base_graph), aux_graph_with_self_loops, treat_epsilons_specially=False)
    setattr(result, "phones", result.labels)
    return result

def compose_with_self_loops(base_graph: k2.Fsa, aux_graph: k2.Fsa) -> k2.Fsa:
    aux_graph_with_self_loops = k2.arc_sort(k2.add_epsilon_self_loops(aux_graph)).to(base_graph.device)
    return k2.compose(base_graph, aux_graph_with_self_loops, treat_epsilons_specially=False, inner_labels="phones")

def intersect_dense_failsafe(**kwargs):
    try:
        return k2.intersect_dense(**kwargs)
    except RuntimeError as e:
        if "Some bad things" not in str(e):
            raise e
        else:
            assert "b_fsas" in kwargs, "k2.intersect_dense failed and there is no b_fsas"
            b_fsas = kwargs.get("b_fsas")
            logging.warning(f"""k2.intersect_dense failed with RuntimeError on device {b_fsas.device}. 
                            All lattices are set trivial.""")
            bs = b_fsas.dim0() if "a_to_b_map" not in kwargs else len(kwargs.get("a_to_b_map"))
            s = "0 1 -1 -1 0.0\n1"
            fsa = k2.Fsa.from_str(s, acceptor=False)
            fsa_vec = k2.create_fsa_vec([fsa.clone() for i in range(bs)])
            fsa_vec.scores = torch.nn.Parameter(torch.full_like(fsa_vec.scores, -float("inf")), requires_grad=b_fsas.scores.requires_grad)
            return fsa_vec.to(b_fsas.device)

def create_sparse_wrapped(indices: List[torch.Tensor],
                          values: torch.Tensor,
                          size: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
                          min_col_index: Optional[int] = None):
    assert size is None or len(indices) == len(size)

    if len(indices) == 2:
        return k2.create_sparse(rows=indices[0],
                                cols=indices[1],
                                values=values,
                                size=size,
                                min_col_index=min_col_index)
    elif len(indices) == 3:
        assert indices[0].ndim == indices[1].ndim == indices[2].ndim == 1
        assert indices[0].numel() == indices[1].numel() == indices[2].numel() == values.numel()

        if min_col_index is not None:
            assert isinstance(min_col_index, int)
            kept_indices = indices[-1] >= min_col_index
            indices = [i[kept_indices] for i in indices]
            values = values[kept_indices]
        if size is not None:
            return torch.sparse_coo_tensor(torch.stack(indices),
                                          values,
                                          size=size,
                                          device=values.device,
                                          requires_grad=values.requires_grad)
        else:
            return torch.sparse_coo_tensor(torch.stack(indices),
                                          values,
                                          device=values.device,
                                          requires_grad=values.requires_grad)
    else:
        raise ValueError(f"len(indices) = {len(indices)}")

def prep_padded_densefsavec(log_softmax: torch.Tensor, supervisions: torch.Tensor,):
    log_softmax_shifted = torch.cat([torch.full((log_softmax.shape[0], log_softmax.shape[1], 1), -float("inf"), device=log_softmax.device), log_softmax], axis=-1)
    log_softmax_padded = torch.zeros((log_softmax_shifted.shape[0], log_softmax_shifted.shape[1] * 2, log_softmax_shifted.shape[2]), device=log_softmax.device)
    log_softmax_padded[:,::2] = log_softmax_shifted
    supervisions_padded = supervisions.clone()
    supervisions_padded[:,2] *= 2
    dense_log_softmax_padded = k2.DenseFsaVec(log_softmax_padded, supervisions_padded)
    return dense_log_softmax_padded

def shift_labels_inpl(lattices: List[k2.Fsa], shift: int):
    for lattice in lattices:
        mask = lattice.labels > 0
        lattice.labels[mask] += shift
        if hasattr(lattice, 'aux_labels'):
            mask = lattice.aux_labels > 0
            lattice.aux_labels[mask] += shift

def get_tot_objf_and_num_frames(
        tot_scores: torch.Tensor,
        reduction: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Figures out the total score(log-prob) over all successful supervision segments
    (i.e. those for which the total score wasn't -infinity).
         Args:
            tot_scores: a Torch tensor of shape (num_segments,) containing total scores
                       from forward-backward
            reduction: a reduction type ('mean', 'sum' or 'none')
        Returns:
             Returns a tuple of 2 scalar tensors: (tot_score, finite_mask)
        where finite_mask is a tensor containing successful segment mask.
    """
    finite_mask = ~torch.isnan(tot_scores) & torch.ne(tot_scores, -float("inf"))
    if reduction == 'mean':
        tot_scores = tot_scores[finite_mask].mean()
    elif reduction == 'sum':
        tot_scores = tot_scores[finite_mask].sum()
    return tot_scores, finite_mask
