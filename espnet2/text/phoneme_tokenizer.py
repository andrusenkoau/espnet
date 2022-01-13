import logging
from pathlib import Path
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
import warnings

import g2p_en
from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


g2p_choices = [
    None,
    "g2p_en",
    "g2p_en_no_space",
    "pyopenjtalk",
    "pyopenjtalk_kana",
    "pyopenjtalk_accent",
    "pyopenjtalk_accent_with_pause",
    "pyopenjtalk_prosody",
    "pypinyin_g2p",
    "pypinyin_g2p_phone",
    "espeak_ng_arabic",
    "espeak_ng_german",
    "espeak_ng_french",
    "espeak_ng_spanish",
    "espeak_ng_russian",
    "espeak_ng_greek",
    "espeak_ng_finnish",
    "espeak_ng_hungarian",
    "espeak_ng_dutch",
    "espeak_ng_english_us_vits",
    "espeak_ng_hindi",
    "g2pk",
    "g2pk_no_space",
    "korean_jaso",
    "korean_jaso_no_space",
]


def split_by_space(text) -> List[str]:
    if "   " in text:
        text = text.replace("   ", " <space> ")
        return [c.replace("<space>", " ") for c in text.split(" ")]
    else:
        return text.split(" ")


def pyopenjtalk_g2p(text) -> List[str]:
    import pyopenjtalk

    # phones is a str object separated by space
    phones = pyopenjtalk.g2p(text, kana=False)
    phones = phones.split(" ")
    return phones


def pyopenjtalk_g2p_accent(text) -> List[str]:
    import pyopenjtalk
    import re

    phones = []
    for labels in pyopenjtalk.run_frontend(text)[1]:
        p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9]+)", labels)
        if len(p) == 1:
            phones += [p[0][0], p[0][2], p[0][1]]
    return phones


def pyopenjtalk_g2p_accent_with_pause(text) -> List[str]:
    import pyopenjtalk
    import re

    phones = []
    for labels in pyopenjtalk.run_frontend(text)[1]:
        if labels.split("-")[1].split("+")[0] == "pau":
            phones += ["pau"]
            continue
        p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9]+)", labels)
        if len(p) == 1:
            phones += [p[0][0], p[0][2], p[0][1]]
    return phones


def pyopenjtalk_g2p_kana(text) -> List[str]:
    import pyopenjtalk

    kanas = pyopenjtalk.g2p(text, kana=True)
    return list(kanas)


def pyopenjtalk_g2p_prosody(text: str, drop_unvoiced_vowels: bool = True) -> List[str]:
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    import pyopenjtalk

    labels = pyopenjtalk.run_frontend(text)[1]
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def pypinyin_g2p(text) -> List[str]:
    from pypinyin import pinyin
    from pypinyin import Style

    phones = [phone[0] for phone in pinyin(text, style=Style.TONE3)]
    return phones


def pypinyin_g2p_phone(text) -> List[str]:
    from pypinyin import pinyin
    from pypinyin import Style
    from pypinyin.style._utils import get_finals
    from pypinyin.style._utils import get_initials

    phones = [
        p
        for phone in pinyin(text, style=Style.TONE3)
        for p in [
            get_initials(phone[0], strict=True),
            get_finals(phone[0], strict=True),
        ]
        if len(p) != 0
    ]
    return phones


class G2p_en:
    """On behalf of g2p_en.G2p.

    g2p_en.G2p isn't pickalable and it can't be copied to the other processes
    via multiprocessing module.
    As a workaround, g2p_en.G2p is instantiated upon calling this class.

    """

    def __init__(self, no_space: bool = False):
        self.no_space = no_space
        self.g2p = None

    def __call__(self, text) -> List[str]:
        if self.g2p is None:
            self.g2p = g2p_en.G2p()

        phones = self.g2p(text)
        if self.no_space:
            # remove space which represents word serapater
            phones = list(filter(lambda s: s != " ", phones))
        return phones


class Wrapper_LexiconG2p:
    """On behalf of LexiconG2p.

    LexiconG2p isn't pickalable and it can't be copied to the other processes
    via multiprocessing module.
    As a workaround, LexiconG2p is instantiated upon calling this class.
    """

    def __init__(
        self,
        lexicon: Union[Path, str],
        no_space: bool = True,
        space_symbol: str = "<space>",
        positional: Union[None, str] = None,
        unk_word: str = "<unk>",
        unk_phon: str = "<spn>",
    ):

        self.lexicon = lexicon
        self.no_space = no_space
        self.space_symbol = space_symbol
        self.positional = positional
        self.unk_word = unk_word
        self.unk_phon = unk_phon
        self.g2p = None

    def __call__(self, text: str) -> List[str]:
        if self.g2p is None:
            self.g2p = LexiconG2p(
                self.lexicon,
                self.no_space,
                self.space_symbol,
                self.positional,
                self.unk_word,
                self.unk_phon,
            )

        return self.g2p(text)


class LexiconG2p:
    """G2p based on user-provided lexicon file."""

    begin_positional_mark = "▁"
    full_positional_begin_mark = "_B"
    full_positional_end_mark = "_E"
    full_positional_inner_mark = "_I"
    full_positional_singular_mark = "_S"

    def _apply_mark_strip(self, token):
        return (
            token.lstrip(self.begin_positional_mark)
            if self.positional == "begin"
            else token.replace(self.full_positional_begin_mark, "")
            .replace(self.full_positional_end_mark, "")
            .replace(self.full_positional_inner_mark, "")
            .replace(self.full_positional_singular_mark, "")
        )

    def __init__(
        self,
        lexicon: Union[Path, str],
        no_space: bool = True,
        space_symbol: str = "<space>",
        positional: Union[None, str] = None,
        unk_word: str = "<unk>",
        unk_phon: str = "<spn>",
    ):
        assert check_argument_types()
        if positional is not None and positional != "begin" and positional != "full":
            raise ValueError(
                f"positional must be one of None, begin, or full: {positional}"
            )
        self.no_space = no_space
        self.space_symbol = space_symbol
        self.positional = positional
        self.unk_word = unk_word
        self.unk_phon = unk_phon
        self.g2p = defaultdict(lambda: [[self.unk_phon]])
        self.p2g = defaultdict(lambda: [self.unk_word])
        if isinstance(lexicon, str):
            lexicon = Path(lexicon)
        with lexicon.open("r", encoding="utf-8") as f:
            for line in f:
                word, trans = line.rstrip().split(" ", 1)
                trans = trans.split()
                # store all possible transcriptions
                # in the order they are in the lexicon
                self.g2p[word].insert(len(self.g2p[word]) - 1, trans)
                # remove all positional marks.
                # It should help in decoding.
                trans = tuple([self._apply_mark_strip(t) for t in trans])
                # store all possible words
                # in the order they are in the lexicon
                self.p2g[trans].insert(len(self.p2g[trans]) - 1, word)

    def __call__(self, text: str) -> List[str]:
        return self.encode(text)

    def encode(self, text: str) -> List[str]:
        # encode words with the first available transcription
        words = text.rstrip().split()
        trans = [] if self.no_space else [self.space_symbol]
        for word in words:
            trans += self.g2p[word][0]
            if not self.no_space:
                trans.append(self.space_symbol)
        return trans

    def decode(self, tokens: Iterable[str]) -> str:
        # decode transcriptions with the first available word
        if self.no_space and self.positional is None or len(tokens) == 0:
            return "".join(tokens)
        word_tran = []
        if self.no_space or tokens[0] != self.space_symbol:
            word_tran.append(self._apply_mark_strip(tokens[0]))
        word_trans = [word_tran]
        if self.no_space:
            for token in tokens[1:]:
                if (
                    self.positional == "begin"
                    and token.startswith(self.begin_positional_mark)
                    or self.positional == "full"
                    and (
                        token.endswith(self.full_positional_begin_mark)
                        or token.endswith(self.full_positional_singular_mark)
                    )
                ):
                    word_tran = [self._apply_mark_strip(token)]
                    word_trans.append(word_tran)
                else:
                    word_tran.append(self._apply_mark_strip(token))
        else:
            for token in tokens[1:]:
                # suppose there are no space_symbols in word transcriptions
                if token == self.space_symbol:
                    word_tran = []
                    word_trans.append(word_tran)
                else:
                    word_tran.append(self._apply_mark_strip(token))
        words = []
        for word_tran in word_trans:
            if len(word_tran) > 0:
                words.append(self.p2g[tuple(word_tran)][0])
        return " ".join(words)


class Phonemizer:
    """Phonemizer module for various languages.

    This is wrapper module of https://github.com/bootphon/phonemizer.
    You can define various g2p modules by specifying options for phonemizer.

    See available options:
        https://github.com/bootphon/phonemizer/blob/master/phonemizer/phonemize.py#L32

    """

    def __init__(
        self,
        backend,
        word_separator: Optional[str] = None,
        syllable_separator: Optional[str] = None,
        phone_separator: Optional[str] = " ",
        strip=False,
        split_by_single_token: bool = False,
        **phonemizer_kwargs,
    ):
        # delayed import
        from phonemizer.backend import BACKENDS
        from phonemizer.separator import Separator

        self.separator = Separator(
            word=word_separator,
            syllable=syllable_separator,
            phone=phone_separator,
        )

        # define logger to suppress the warning in phonemizer
        logger = logging.getLogger("phonemizer")
        logger.setLevel(logging.ERROR)
        self.phonemizer = BACKENDS[backend](
            **phonemizer_kwargs,
            logger=logger,
        )
        self.strip = strip
        self.split_by_single_token = split_by_single_token

    def __call__(self, text) -> List[str]:
        tokens = self.phonemizer.phonemize(
            [text],
            separator=self.separator,
            strip=self.strip,
            njobs=1,
        )[0]
        if not self.split_by_single_token:
            return tokens.split()
        else:
            # "a: ab" -> ["a", ":", "<space>",  "a", "b"]
            # TODO(kan-bayashi): space replacement should be dealt in PhonemeTokenizer
            return [c.replace(" ", "<space>") for c in tokens]


class PhonemeTokenizer(AbsTokenizer):
    def __init__(
        self,
        g2p_type: Union[None, str],
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
        g2p_lexicon_path: Union[Path, str] = None,
        g2p_lexicon_conf: Dict = None,
    ):
        assert check_argument_types()
        if g2p_type is None:
            self.g2p = split_by_space
        elif g2p_type == "g2p_en":
            self.g2p = G2p_en(no_space=False)
        elif g2p_type == "g2p_en_no_space":
            self.g2p = G2p_en(no_space=True)
        elif g2p_type == "pyopenjtalk":
            self.g2p = pyopenjtalk_g2p
        elif g2p_type == "pyopenjtalk_kana":
            self.g2p = pyopenjtalk_g2p_kana
        elif g2p_type == "pyopenjtalk_accent":
            self.g2p = pyopenjtalk_g2p_accent
        elif g2p_type == "pyopenjtalk_accent_with_pause":
            self.g2p = pyopenjtalk_g2p_accent_with_pause
        elif g2p_type == "pyopenjtalk_prosody":
            self.g2p = pyopenjtalk_g2p_prosody
        elif g2p_type == "pypinyin_g2p":
            self.g2p = pypinyin_g2p
        elif g2p_type == "pypinyin_g2p_phone":
            self.g2p = pypinyin_g2p_phone
        elif g2p_type == "espeak_ng_arabic":
            self.g2p = Phonemizer(language="ar", backend="espeak", with_stress=True)
        elif g2p_type == "g2p_lexicon":
            if g2p_lexicon_path is None:
                raise ValueError(
                    f"g2p_lexicon_path is required for g2p_lexicon g2p_type:"
                    f" {g2p_lexicon_path}, {g2p_type}"
                )
            if space_symbol != g2p_lexicon_conf["space_symbol"]:
                raise ValueError(
                    f"space_symbol and g2p_lexicon_conf.space_symbol must match:"
                    f" {space_symbol}, {g2p_lexicon_conf['space_symbol']}"
                )
            self.g2p = Wrapper_LexiconG2p(g2p_lexicon_path, **g2p_lexicon_conf)
        else:
            raise NotImplementedError(f"Not supported: g2p_type={g2p_type}")

        self.g2p_type = g2p_type
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'g2p_type="{self.g2p_type}", '
            f'space_symbol="{self.space_symbol}", '
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            ")"
        )

    def text2tokens(self, line: str) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                tokens.append(t)
                line = line[1:]

        line = "".join(tokens)
        tokens = self.g2p(line)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        # phoneme type is not invertible
        return (
            self.g2p.decode(tokens)
            if self.g2p_type == "g2p_lexicon"
            else "".join(tokens)
        )
