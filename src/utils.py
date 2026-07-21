from fractions import Fraction
from typing import Set, Union

import torch


def get_dtype(data_type: str) -> torch.dtype:
    data_type2dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128
    }
    return data_type2dtype[data_type]


def get_matmul_tensor(n: int, m: int, p: int, device: str) -> torch.Tensor:
    tensor = torch.zeros((n * m, m * p, p * n), dtype=torch.float64, device=device)

    for i in range(n):
        for j in range(p):
            for k in range(m):
                tensor[i * m + k, k * p + j, j * n + i] = 1.0

    return tensor


def normalize_value(value: Union[int, float, complex]) -> Union[int, float, complex]:
    if isinstance(value, float) and value == int(value):
        return int(value)

    if isinstance(value, complex):
        if value.imag == 0:
            return normalize_value(value.real)

        if value.real == 0:
            return normalize_value(value.imag) * 1j

        return normalize_value(value.real) + normalize_value(value.imag) * 1j

    return value


def value2str(value: Union[int, float, complex, Fraction]) -> str:
    if isinstance(value, int):
        return str(value)

    if isinstance(value, Fraction):
        return str(value.numerator) if value.denominator == 1 else f'"{value.numerator}/{value.denominator}"'

    if isinstance(value, float):
        return value2str(Fraction(value))

    re, im = Fraction(value.real), Fraction(value.imag)

    if im == 0:
        return value2str(re)

    if re == 0:
        return f'"{im.numerator}i"' if re.denominator == 1 else f'"{re.numerator}/{re.denominator}i"'

    re_str = str(re.numerator) if re.denominator == 1 else f"{re.numerator}/{re.denominator}"
    im_str = f"{im.numerator:+}i" if im.denominator == 1 else f"{im.numerator:+}/{im.denominator}i"
    return f'"{re_str}{im_str}"'


def get_values_ring(values: Set[Union[int, float, complex]]) -> str:
    have_fractions = False
    have_integers = False

    for value in values:
        if isinstance(value, complex):
            if value.imag != 0:
                return "C"

            value = value.real

        value = Fraction(value)
        if value.denominator > 1:
            have_fractions = True
        elif abs(value.numerator) > 1:
            have_integers = True

    if have_fractions:
        return "Q"

    if have_integers:
        return "Z"

    return "ZT"
