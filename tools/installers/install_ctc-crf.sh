#!/bin/bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# TODO(kamo): Consider clang case
# Note: Requires gcc>=4.9.2 to build extensions with pytorch>=1.0
if python3 -c 'import torch as t;assert t.__version__[0] == "1"' &> /dev/null; then \
    python3 -c "from distutils.version import LooseVersion as V;assert V('$(gcc -dumpversion)') >= V('4.9.2'), 'Requires gcc>=4.9.2'"; \
fi

rm -rf CAT
git clone https://github.com/thu-spmi/CAT.git

(
    set -euo pipefail
    cd CAT
    (
        set -euo pipefail
        cp -f src/kaldi-patch/latgen-faster.cc kaldi/src/bin
        cd kaldi/src/bin
        sed -i "s:BINFILES =:BINFILES = latgen-faster:g" Makefile
        make latgen-faster
        cd ../../../
    )

    cd src/ctc_crf
    (
        set -euo pipefail
        OPENFST=../../kaldi/tools/openfst make GPUCTC GPUDEN PATHWEIGHT
    )

    (
        set -euo pipefail
        python setup_1_0.py install
    )
)
