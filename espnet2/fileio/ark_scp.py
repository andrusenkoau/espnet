import collections.abc
from pathlib import Path
from typing import Union

import kaldiio
import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text


def _load_kaldi(retval):
    if isinstance(retval, tuple):
        assert len(retval) == 2, len(retval)
        if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
            # sound scp case
            rate, array = retval
        elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
            # Extended ark format case
            array, rate = retval
        else:
            raise RuntimeError(f"Unexpected type: {type(retval[0])}, {type(retval[1])}")

        # Multichannel wave fie
        # array: (NSample, Channel) or (Nsample)

    else:
        # Normal ark case
        assert isinstance(retval, np.ndarray), type(retval)
        array = retval
    return array


class ArkScpWriter:
    """Writer class for a scp file of kaldi mat or ark file.

    Examples:
        key1 /some/path/a.mat
        key2 /some/path/b.mat
        key3 /some/path/c.mat
        key4 /some/path/d.mat
        ...

        >>> writer = ArkScpWriter('./data/', './data/feat.scp')
        >>> writer['aa'] = numpy_array
        >>> writer['bb'] = numpy_array

    """

    def __init__(
        self, outdir: Union[Path, str], scpfile: Union[Path, str], save_mat=True
    ):
        assert check_argument_types()
        self.save_mat = save_mat
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        if self.save_mat:
            self.fscp = scpfile.open("w", encoding="utf-8")
        else:
            self.fscp = scpfile
            self.fark = Path(self.dir, self.fscp.name.rstrip(self.fscp.suffix) + ".ark")

        self.data = {}

    def get_path(self, key):
        if self.save_mat:
            return self.data[key]
        else:
            raise NotImplementedError()

    def __setitem__(self, key, value):
        assert isinstance(value, np.ndarray), type(value)
        if self.save_mat:
            p = self.dir / f"{key}.mat"
            p.parent.mkdir(parents=True, exist_ok=True)
            kaldiio.save_mat(str(p), value)
            self.fscp.write(f"{key} {p}\n")

            # Store the file path
            self.data[key] = str(p)
        else:
            kaldiio.save_ark(
                str(self.fark), {key: value}, append=True, scp=str(self.fscp)
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.save_mat:
            self.close()

    def close(self):
        self.fscp.close()


class ArkScpReader(collections.abc.Mapping):
    """Reader class for a scp file of numpy file.

    Examples:
        key1 /some/path/a.mat
        key2 /some/path/b.mat
        key3 /some/path/c.mat
        key4 /some/path/d.mat
        ...

        >>> reader = ArkScpReader('feat.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str], read_mat=True):
        assert check_argument_types()
        self.read_mat = read_mat
        self.fname = Path(fname)
        self.data = (
            read_2column_text(self.fname)
            if self.read_mat
            else kaldiio.load_scp(self.fname)
        )

    def get_path(self, key):
        if self.read_mat:
            return self.data[key]
        else:
            raise NotImplementedError()

    def __getitem__(self, key) -> np.ndarray:
        p = self.data[key]
        return _load_kaldi(p) if self.read_mat else kaldiio.load_mat(p)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
