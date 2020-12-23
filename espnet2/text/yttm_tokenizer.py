from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

import youtokentome as yttm
from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


class YttmTokenizer(AbsTokenizer):
    def __init__(
        self,
        model: Union[Path, str],
        dropout_prob: float = 0.0
    ):
        assert check_argument_types()
        self.model = str(model)
        self.dropout_prob = dropout_prob
        # NOTE(kamo):
        # Don't build SentencePieceProcessor in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.yttm = None

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def text2tokens(self, line: str, dropout_prob: float = 0.0) -> List[str]:
        self.yttm = yttm.BPE(model=self.model)
        return self.yttm.encode([line], output_type=yttm.OutputType.SUBWORD, dropout_prob=dropout_prob)[0]

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return "".join(tokens).replace('▁', ' ')
