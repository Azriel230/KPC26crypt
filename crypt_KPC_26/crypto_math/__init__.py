"""crypto_math — учебная библиотека арифметики для асимметричной криптографии.

Пакет пополняется в следующих ноутбуках курса. Сейчас содержит модуль
:mod:`crypto_math.number_theory`.
"""

from .number_theory import (
    random_exact_bits,
    crt,
    crt_encode,
    divides,
    extended_gcd,
    garner_crt,
    gcd,
    mod_inverse,
    mod_pow,
    solve_linear,
    solve_linear_diophantine,
    is_prime_fast,
    primes,
    next_prime,
    smallest_factor,
    decompose,
    recompose,
    euler_phi,
    multiplicative_order,
    is_primitive_root,
    primitive_root
)

__all__ = [
    "random_exact_bits",
    "divides",
    "gcd",
    "extended_gcd",
    "solve_linear_diophantine",
    "mod_inverse",
    "solve_linear",
    "mod_pow",
    "crt_encode",
    "crt",
    "garner_crt",
    "is_prime_fast",
    "primes",
    "next_prime",
    "smallest_factor",
    "decompose",
    "recompose",
    "euler_phi",
    "multiplicative_order",
    "is_primitive_root",
    "primitive_root"
]
