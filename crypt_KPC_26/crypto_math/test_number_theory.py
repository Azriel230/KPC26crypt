"""Property-based тесты для crypto_math.number_theory.

Готовые реализации (math.gcd, pow(a, -1, n), трёхаргументный pow) используются
здесь только как независимый эталон. Коэффициенты Безу и частные решения не
единственны, поэтому проверяются свойства, а не заранее выбранные ответы.

Запуск из корня проекта: python -m pytest -q
Необходимые пакеты: pip install pytest hypothesis
"""

import math

import pytest
from hypothesis import assume, given, strategies as st

from crypto_math.number_theory import (
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
)

nonzero_pairs = st.tuples(st.integers(), st.integers()).filter(lambda p: p != (0, 0))


# --------------------------------------------------------------------------- #
# divides, gcd, extended_gcd
# --------------------------------------------------------------------------- #

@given(st.integers(), st.integers())
def test_divides_matches_definition(a, b):
    assert divides(a, b) == (b == 0 if a == 0 else b % a == 0)


@given(nonzero_pairs)
def test_gcd_matches_reference(pair):
    a, b = pair
    assert gcd(a, b) == math.gcd(a, b)


@given(nonzero_pairs)
def test_extended_gcd_bezout(pair):
    a, b = pair
    g, x, y = extended_gcd(a, b)
    assert g == math.gcd(a, b)
    assert a * x + b * y == g


def test_gcd_rejects_zero_zero():
    with pytest.raises(ValueError):
        gcd(0, 0)


def test_extended_gcd_rejects_zero_zero():
    with pytest.raises(ValueError):
        extended_gcd(0, 0)


# --------------------------------------------------------------------------- #
# solve_linear_diophantine: a*x + b*y = c
# --------------------------------------------------------------------------- #

def test_diophantine_rejects_degenerate_case():
    with pytest.raises(ValueError):
        solve_linear_diophantine(0, 0, 5)


@given(nonzero_pairs, st.integers())
def test_diophantine_solvable_iff_gcd_divides_c(pair, c):
    a, b = pair
    g = math.gcd(a, b)
    result = solve_linear_diophantine(a, b, c)
    assert (result is None) == (c % g != 0)


@given(nonzero_pairs, st.integers(), st.integers(min_value=-10**6, max_value=10**6))
def test_diophantine_parametrization_gives_solutions(pair, k, t):
    """Для разрешимого уравнения (c = k*g) все точки (x0+dx*t, y0+dy*t) — решения."""
    a, b = pair
    g = math.gcd(a, b)
    c = k * g
    result = solve_linear_diophantine(a, b, c)
    assert result is not None
    x0, y0, dx, dy = result
    assert dx == b // g and dy == -a // g
    assert a * x0 + b * y0 == c
    assert a * (x0 + dx * t) + b * (y0 + dy * t) == c


@given(nonzero_pairs, st.integers(), st.integers())
def test_diophantine_family_is_complete(pair, x, y):
    """Любое известное решение (x, y) лежит в семействе (1.20) при некотором t."""
    a, b = pair
    c = a * x + b * y  # (x, y) — решение по построению
    result = solve_linear_diophantine(a, b, c)
    assert result is not None
    x0, y0, dx, dy = result
    if dx != 0:
        assert (x - x0) % dx == 0
        t = (x - x0) // dx
    else:  # b == 0, тогда dy = -a/g = ±1
        assert x == x0
        t = (y - y0) // dy
    assert (x, y) == (x0 + dx * t, y0 + dy * t)


@pytest.mark.parametrize(
    "a, b, c, solvable",
    [(20, 50, 510, True), (72, 100, 17, False), (3, 4, -1, True)],
)
def test_diophantine_examples_from_notebook(a, b, c, solvable):
    result = solve_linear_diophantine(a, b, c)
    assert (result is not None) == solvable
    if result is not None:
        x0, y0, dx, dy = result
        for t in (-3, -1, 0, 1, 3):
            assert a * (x0 + dx * t) + b * (y0 + dy * t) == c


# --------------------------------------------------------------------------- #
# solve_linear: a*x ≡ b (mod n)
# --------------------------------------------------------------------------- #

def test_solve_linear_notebook_examples():
    assert solve_linear(4, 2, 10) == [3, 8]
    assert solve_linear(2, 1, 4) == []
    assert solve_linear(27, 4, 25) == [2]


@given(st.integers(), st.integers(), st.integers(max_value=1))
def test_solve_linear_rejects_small_modulus(a, b, n):
    with pytest.raises(ValueError):
        solve_linear(a, b, n)


@given(
    st.integers(min_value=-10**9, max_value=10**9),
    st.integers(min_value=-10**9, max_value=10**9),
    st.integers(min_value=2, max_value=10**9),
)
def test_solve_linear_contract(a, b, n):
    """Постусловие: отсортированные различные решения из [0, n), каждое верно;
    решений ровно gcd(a, n), если gcd(a, n) | b, и ни одного иначе."""
    g = math.gcd(a, n)
    assume(g <= 1000)  # ограничиваем размер возвращаемого списка
    xs = solve_linear(a, b, n)
    assert xs == sorted(set(xs))
    assert all(0 <= x < n for x in xs)
    assert all((a * x - b) % n == 0 for x in xs)
    assert len(xs) == (g if b % g == 0 else 0)


@given(st.integers(min_value=2, max_value=60), st.integers(), st.integers())
def test_solve_linear_is_complete_for_small_modulus(n, a, b):
    """Полнота: перебор допустим только в тесте, как эталон для малых модулей."""
    expected = [x for x in range(n) if (a * x - b) % n == 0]
    assert solve_linear(a, b, n) == expected


@given(st.integers(), st.integers(), st.integers(min_value=2))
def test_solve_linear_coprime_case_is_unique_inverse_multiple(a, b, n):
    """При gcd(a, n) = 1 решение единственно и равно b * a^{-1} mod n."""
    assume(math.gcd(a, n) == 1)
    assert solve_linear(a, b, n) == [(b * pow(a, -1, n)) % n]


@given(
    st.integers(min_value=2, max_value=40),
    st.integers(min_value=1, max_value=10**6),
    st.integers(min_value=-10**6, max_value=10**6),
)
def test_solve_linear_many_solutions(g, m, j):
    """Случай gcd(a, n) > 1: при G | b решений ровно G >= g, все различны и верны."""
    n, a = g * (m + 1), 3 * g
    G = math.gcd(a, n)
    assert G >= g
    b = G * j  # разрешимо по построению
    xs = solve_linear(a, b, n)
    assert len(xs) == G
    assert xs == sorted(set(xs))
    assert all((a * x - b) % n == 0 for x in xs)


# --------------------------------------------------------------------------- #
# mod_inverse, mod_pow
# --------------------------------------------------------------------------- #

@given(st.integers(), st.integers(min_value=2))
def test_mod_inverse_contract(a, n):
    if math.gcd(a, n) == 1:
        inv = mod_inverse(a, n)
        assert 0 <= inv < n and (a * inv) % n == 1
        assert inv == pow(a, -1, n)
    else:
        with pytest.raises(ValueError):
            mod_inverse(a, n)


@given(st.integers(), st.integers(max_value=1))
def test_mod_inverse_rejects_small_modulus(a, n):
    with pytest.raises(ValueError):
        mod_inverse(a, n)


@given(st.integers(), st.integers(min_value=0, max_value=2**64), st.integers(min_value=1))
def test_mod_pow_matches_builtin(a, e, n):
    assert mod_pow(a, e, n) == pow(a, e, n)


@given(st.integers(), st.integers(min_value=0, max_value=2**64), st.integers(min_value=1))
def test_mod_pow_result_in_range(a, e, n):
    assert 0 <= mod_pow(a, e, n) < n


def test_mod_pow_rejects_bad_arguments():
    with pytest.raises(ValueError):
        mod_pow(2, 3, 0)
    with pytest.raises(ValueError):
        mod_pow(2, -1, 7)


# --------------------------------------------------------------------------- #
# crt_encode, crt, garner_crt
# --------------------------------------------------------------------------- #

MODULI = [3, 5, 7, 11, 13, 17]


@given(st.integers())
def test_crt_roundtrip(x):
    N = math.prod(MODULI)
    residues = crt_encode(x, MODULI)
    assert residues == [x % m for m in MODULI]
    assert crt(residues, MODULI) == garner_crt(residues, MODULI) == x % N


@given(st.lists(st.integers(), min_size=6, max_size=6))
def test_crt_satisfies_every_congruence(residues):
    x = crt(residues, MODULI)
    assert 0 <= x < math.prod(MODULI)
    assert all(x % m == a % m for a, m in zip(residues, MODULI))
    assert x == garner_crt(residues, MODULI)


def test_crt_classical_example():
    assert crt([2, 3, 2], [3, 5, 7]) == 23
    assert garner_crt([2, 3, 2], [3, 5, 7]) == 23


@pytest.mark.parametrize("func", [crt, garner_crt])
def test_crt_rejects_invalid_input(func):
    with pytest.raises(ValueError):
        func([1, 2], [6, 9])      # не взаимно просты
    with pytest.raises(ValueError):
        func([1], [3, 5])         # разная длина
    with pytest.raises(ValueError):
        func([], [])              # пустой вход
    with pytest.raises(ValueError):
        func([0, 0], [1, 5])      # модуль < 2


def test_crt_encode_rejects_invalid_input():
    with pytest.raises(ValueError):
        crt_encode(5, [])
    with pytest.raises(ValueError):
        crt_encode(5, [1, 7])
