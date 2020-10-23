#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Repeat the same layer definition."""

import torch


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""

    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args


class MultiSequentialArg2(torch.nn.Sequential):
    """2-input 2-output torch.nn.Sequential."""

    def forward(self, input1, input2):
        """Repeat."""
        for m in self:
            input1, input2 = m(input1, input2)
        return input1, input2


class MultiSequentialArg3(torch.nn.Sequential):
    """2-input 2-output torch.nn.Sequential."""

    def forward(self, input1, input2, input3):
        """Repeat."""
        for m in self:
            input1, input2, input3 = m(input1, input2, input3)
        return input1, input2, input3


class MultiSequentialArg4(torch.nn.Sequential):
    """2-input 2-output torch.nn.Sequential."""

    def forward(self, input1, input2, input3, input4):
        """Repeat."""
        for m in self:
            input1, input2, input3, input4 = m(input1, input2, input3, input4)
        return input1, input2, input3, input4


def repeat(N, fn):
    """Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.

    Returns:
        MultiSequential: Repeated model instance.

    """
    return MultiSequential(*[fn(n) for n in range(N)])
