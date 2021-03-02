from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import sentencepiece as spm
from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


class SentencepiecesTokenizer(AbsTokenizer):
    def __init__(
        self,
        model: Union[Path, str],
        bpe_alpha: float = 0.0,
        replace_position_mark: Optional[str] = None,
    ):
        assert check_argument_types()
        assert replace_position_mark != "▁"
        self.model = str(model)
        self.bpe_alpha = bpe_alpha
        self.replace_position_mark = replace_position_mark
        # NOTE(kamo):
        # Don't build SentencePieceProcessor in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.sp = None

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_sentence_piece_processor(self):
        # Build SentencePieceProcessor lazily.
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model)

    def text2tokens(self, line: str) -> List[str]:
        self._build_sentence_piece_processor()
        if self.bpe_alpha == 0.0:
            encoded = self.sp.SampleEncodeAsPieces(
                line, nbest_size=1, alpha=self.bpe_alpha
            )
        else:
            # Do not allow to separate "▁" piece
            encoded = (
                " ".join(
                    self.sp.SampleEncodeAsPieces(
                        line, nbest_size=-1, alpha=self.bpe_alpha
                    )
                )
                .replace("▁ ", "▁")
                .split()
            )
        return (
            encoded
            if self.replace_position_mark is None
            else " ".join(encoded + [self.replace_position_mark])
            .replace("▁", self.replace_position_mark)
            .strip()
            .split()
        )

    def tokens2text(self, tokens: Iterable[str]) -> str:
        self._build_sentence_piece_processor()
        return self.sp.DecodePieces(list(tokens))
