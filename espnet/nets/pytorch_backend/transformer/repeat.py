#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Repeat the same layer definition."""

import torch
from torch.utils.checkpoint import checkpoint


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args


class MultiSequentialCheckpointed(MultiSequential):
    """Checkpointed MultiSequential."""

    def forward(self, *args):
        """Repeat."""
        if self.training:
            for m in self:
                args = checkpoint(m, *args)
                args = (args[0], *[arg.detach() for arg in args[1:]])
            return args
        else:
            return super().forward(*args)


class MultiSequentialArg2(torch.nn.Sequential):
    """2-input 2-output torch.nn.Sequential."""

    def forward(self, input1, input2):
        """Repeat."""
        for m in self:
            input1, input2 = m(input1, input2)
        return input1, input2


class MultiSequentialArg3(torch.nn.Sequential):
    """3-input 3-output torch.nn.Sequential."""

    def forward(self, input1, input2, input3):
        """Repeat."""
        for i, m in enumerate(self):
            input1, input2, input3 = m(input1, input2, input3)
        return input1, input2, input3


class MultiSequentialArg4(torch.nn.Sequential):
    """4-input 4-output torch.nn.Sequential."""

    def forward(self, input1, input2, input3, input4):
        """Repeat."""
        for m in self:
            input1, input2, input3, input4 = m(input1, input2, input3, input4)
        return input1, input2, input3, input4


def repeat(N, fn, checkpointed=False):
    """Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.

    Returns:
        MultiSequential: Repeated model instance.

    """
    if checkpointed:
        return MultiSequentialCheckpointed(*[fn(n) for n in range(N)])
    return MultiSequential(*[fn(n) for n in range(N)])
