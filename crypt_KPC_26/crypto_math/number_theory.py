"""Элементарная теория чисел для асимметричной криптографии.

Модуль содержит проверенные реализации операций ноутбука 1:

    divides, gcd, extended_gcd, solve_linear_diophantine, mod_inverse,
    solve_linear, mod_pow, crt_encode, crt, garner_crt.

Граф зависимостей::

    gcd -> extended_gcd -> solve_linear_diophantine
                        -> mod_inverse -> crt, garner_crt
                        -> solve_linear
    mod_pow (отдельная ветвь: двоичное представление показателя)

Внутри модуля намеренно не используются готовые реализации изучаемых
алгоритмов (``math.gcd``, ``pow(a, -1, n)``, трёхаргументный ``pow``,
``sympy``, Sage): они допустимы только в тестах как независимый эталон.
"""

from __future__ import annotations
import math
from math import isqrt
import secrets


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
]



def random_exact_bits(bits: int) -> int:
    """Возвращает положительное целое ровно заданной битовой длины."""
    if bits < 1:
        raise ValueError("bits должно быть положительным")

    n = secrets.randbits(bits)
    n |= 1 << (bits - 1)
    return n



# --------------------------------------------------------------------------- #
# Делимость и НОД
# --------------------------------------------------------------------------- #

def divides(a: int, b: int) -> bool:
    """Проверяет делимость ``a | b`` (Определение 1.1).

    Контракт:
        Вход: любые целые ``a``, ``b``.
        Выход: ``True`` тогда и только тогда, когда существует целое ``k``
        такое, что ``b == a * k``.

    Крайний случай: ``0 | b`` тогда и только тогда, когда ``b == 0``.
    """
    if a == 0:
        return b == 0
    return b % a == 0


def gcd(a: int, b: int) -> int:
    """Наибольший общий делитель алгоритмом Евклида.

    Контракт:
        Вход: целые ``a``, ``b``, не равные нулю одновременно.
        Выход: положительное ``d = gcd(a, b)``, такое что ``d | a``, ``d | b``
        и любой общий делитель ``a`` и ``b`` делит ``d``.
        Принято ``gcd(a, 0) == |a|``.

    Инвариант цикла: ``gcd(a, b) == gcd(b, a mod b)`` (формула 1.13).
    Сложность: ``O(L)`` делений для ``L``-битных входов (теорема Ламе).

    Raises:
        ValueError: при ``a == b == 0`` (по соглашению курса НОД не определён).
    """
    if a == 0 and b == 0:
        raise ValueError("gcd(0, 0) is undefined by the course convention")

    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Расширенный алгоритм Евклида: НОД и коэффициенты Безу.

    Контракт:
        Вход: целые ``a``, ``b``, не равные нулю одновременно.
        Выход: тройка ``(g, x, y)``, для которой одновременно
        ``g == gcd(a, b)`` и ``a * x + b * y == g``.

    Коэффициенты Безу не единственны, поэтому корректность проверяется
    свойством ``a*x + b*y == g``, а не сравнением с заранее выбранной парой.

    Инвариант (1.15): на каждом шаге ``old_r = old_s*|a| + old_t*|b|`` и
    ``r = s*|a| + t*|b|``. Знаки ``a`` и ``b`` восстанавливаются после цикла.
    Сложность: ``O(L^2)`` битовых операций в классической модели.

    Raises:
        ValueError: при ``a == b == 0``.
    """
    if a == 0 and b == 0:
        raise ValueError("gcd(0, 0) is undefined by the course convention")

    A, B = abs(a), abs(b)

    old_r, r = A, B
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        # Инварианты (1.15) — документируют доказательство корректности.
        assert old_r == old_s * A + old_t * B
        assert r == s * A + t * B

        q = old_r // r

        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    assert old_r == old_s * A + old_t * B

    x = old_s if a >= 0 else -old_s
    y = old_t if b >= 0 else -old_t
    return old_r, x, y


# --------------------------------------------------------------------------- #
# Линейные диофантовы уравнения и сравнения
# --------------------------------------------------------------------------- #

def solve_linear_diophantine(a: int, b: int, c: int) -> tuple[int, int, int, int] | None:
    """Решает линейное диофантово уравнение ``a*x + b*y = c`` (теорема 1.3).

    Контракт:
        Вход: целые ``a``, ``b``, ``c``; ``a`` и ``b`` не равны нулю одновременно.
        Выход:
            * ``None``, если ``gcd(a, b)`` не делит ``c`` (решений нет);
            * иначе кортеж ``(x0, y0, dx, dy)``, где ``a*x0 + b*y0 == c``,
              ``dx == b // g``, ``dy == -a // g``, ``g = gcd(a, b)``.

    Постусловие: множество всех целочисленных решений есть
    ``{(x0 + dx*t, y0 + dy*t) : t in Z}`` (формула 1.20).

    Raises:
        ValueError: при ``a == b == 0`` (вырожденный случай).
    """
    if a == 0 and b == 0:
        raise ValueError("degenerate equation: a == b == 0")

    g, u, v = extended_gcd(a, b)
    if c % g != 0:
        return None

    k = c // g
    return u * k, v * k, b // g, -a // g


def solve_linear(a: int, b: int, n: int) -> list[int]:
    """Решает линейное сравнение ``a*x ≡ b (mod n)`` (теорема 1.4).

    Контракт:
        Вход: целые ``a``, ``b`` и модуль ``n >= 2``.
        Выход: отсортированный по возрастанию список всех различных решений
        ``x`` в диапазоне ``0 <= x < n``; пустой список ``[]``, если решений нет.

    Постусловие: пусть ``g = gcd(a, n)``. Решения существуют тогда и только
    тогда, когда ``g | b``; в этом случае их ровно ``g`` и они имеют вид
    ``x0 + k * (n // g)``, ``k = 0, ..., g - 1`` (формула 1.32).

    Raises:
        ValueError: при ``n < 2``.
    """
    if n < 2:
        raise ValueError("modulus must be >= 2")

    g, u, _ = extended_gcd(a, n)  # a*u + n*v == g
    if b % g != 0:
        return []

    step = n // g
    # Приведённое сравнение (a/g) x ≡ (b/g) (mod n/g) имеет единственный
    # класс решений x0 = u * (b/g) mod n/g.
    x0 = (u * (b // g)) % step
    return [x0 + k * step for k in range(g)]


# --------------------------------------------------------------------------- #
# Обратный элемент и возведение в степень
# --------------------------------------------------------------------------- #

def mod_inverse(a: int, modulus: int) -> int:
    """Обратный к ``a`` элемент по модулю ``modulus`` (критерий 1.23).

    Контракт:
        Вход: целое ``a`` и ``modulus >= 2``.
        Выход: ``x`` с ``0 <= x < modulus`` и ``(a * x) % modulus == 1``,
        если ``gcd(a, modulus) == 1``.

    Raises:
        ValueError: при ``modulus < 2`` (нарушение программного контракта)
            или если ``gcd(a, modulus) != 1`` (обратного элемента не существует).
    """
    if modulus < 2:
        raise ValueError("modulus must be >= 2")

    g, x, _ = extended_gcd(a, modulus)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % modulus


def mod_pow(base: int, exponent: int, modulus: int) -> int:
    """Быстрое модульное возведение в степень (right-to-left, §1.6).

    Контракт:
        Вход: целое ``base``, ``exponent >= 0``, ``modulus > 0``.
        Выход: целое ``r`` с ``0 <= r < modulus`` и
        ``r ≡ base**exponent (mod modulus)``.

    Инвариант цикла: ``result * base**e ≡ base_0**exponent (mod modulus)``.
    Сложность: ``O(log e)`` модульных умножений; ``O(E * L^2)`` битовых операций.

    Не выполняется за постоянное время: число операций зависит от битов
    показателя, поэтому для секретных показателей функция непригодна.

    Raises:
        ValueError: при ``modulus <= 0`` или ``exponent < 0``.
    """
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    if exponent < 0:
        raise ValueError("this implementation expects exponent >= 0")

    result = 1 % modulus
    base %= modulus
    e = exponent

    while e:
        if e & 1:
            result = (result * base) % modulus
        base = (base * base) % modulus
        e >>= 1

    return result


# --------------------------------------------------------------------------- #
# Китайская теорема об остатках
# --------------------------------------------------------------------------- #

def _product(values: list[int]) -> int:
    """Произведение элементов списка (пустое произведение равно 1)."""
    result = 1
    for v in values:
        result *= v
    return result


def _check_crt_input(residues: list[int], moduli: list[int]) -> None:
    """Проверяет контракт входа ``crt`` / ``garner_crt``.

    Raises:
        ValueError: если списки пусты или разной длины, какой-либо модуль
            меньше 2 либо модули не попарно взаимно просты.
    """
    if len(residues) != len(moduli) or not residues:
        raise ValueError("residues and moduli must have equal non-zero length")
    if any(n < 2 for n in moduli):
        raise ValueError("all moduli must be >= 2")

    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                raise ValueError("moduli must be pairwise coprime")


def crt_encode(x: int, moduli: list[int]) -> list[int]:
    """Переход global → local: остатки ``x`` по каждому модулю.

    Контракт:
        Вход: целое ``x`` и непустой список модулей, каждый ``>= 2``.
        Выход: список ``[x mod n_1, ..., x mod n_k]``.

    Raises:
        ValueError: если список модулей пуст или какой-либо модуль меньше 2.
    """
    if not moduli:
        raise ValueError("moduli must be non-empty")
    if any(n < 2 for n in moduli):
        raise ValueError("all moduli must be >= 2")
    return [x % n for n in moduli]


def crt(residues: list[int], moduli: list[int]) -> int:
    """Китайская теорема об остатках: базисные элементы (формула 1.45).

    Контракт:
        Вход: непустые списки одинаковой длины; все модули ``>= 2`` и попарно
        взаимно просты.
        Выход: наименьший неотрицательный представитель ``x`` единственного
        класса решений по модулю ``N = n_1 * ... * n_k``, то есть
        ``0 <= x < N`` и ``x ≡ a_i (mod n_i)`` для каждого ``i``.

    Свойство (1.50): ``crt(crt_encode(x, moduli), moduli) == x % N``.

    Raises:
        ValueError: при нарушении контракта входа.
    """
    _check_crt_input(residues, moduli)

    N = _product(moduli)
    x = 0

    for a_i, n_i in zip(residues, moduli):
        N_i = N // n_i
        M_i = mod_inverse(N_i, n_i)
        x += a_i * N_i * M_i  # a_i * e_i, e_i = N_i * M_i

    return x % N


def garner_crt(residues: list[int], moduli: list[int]) -> int:
    """Последовательная реконструкция КТО: алгоритм Гарнера (формула 1.51).

    Контракт:
        Тот же, что у :func:`crt`: возвращает наименьший неотрицательный
        представитель класса решений по модулю ``N``.

    Инвариант: после обработки ``i`` модулей ``x ≡ a_j (mod m_j)`` для
    ``j <= i`` и ``0 <= x < m_1 * ... * m_i``.

    Raises:
        ValueError: при нарушении контракта входа.
    """
    _check_crt_input(residues, moduli)

    x = residues[0] % moduli[0]
    M = moduli[0]

    for a_i, m_i in zip(residues[1:], moduli[1:]):
        t = ((a_i - x) * mod_inverse(M, m_i)) % m_i
        x += t * M
        M *= m_i

    return x


def is_prime_fast(n: int) -> bool:
    """Быстрая проверка числа на простоту методом перебора делителей.

    Контракт:
        Принимает целое число ``n``. Возвращает ``True``, если ``n`` является
        простым числом, и ``False`` в противном случае.

    Оптимизация:
        После проверки на чётность (и особых случаев 0, 1, 2) перебор
        ведётся только по нечётным числам ``3, 5, 7, ...`` до ``sqrt(n)``
        включительно. Это сокращает число итераций в два раза.

    Инвариант: если у числа ``n`` есть делитель ``d > 1``, то существует
        и делитель ``n // d <= sqrt(n)``, поэтому перебора до ``sqrt(n)``
        достаточно для корректного ответа.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    limit = math.isqrt(n)
    # Перебираем только нечетные делители
    for i in range(3, limit + 1, 2):
        if n % i == 0:
            return False
    return True


def primes(n: int) -> list[int]:
    """Возвращает список всех простых чисел p <= n (решето Эратосфена).

    Контракт:
        Принимает целое число ``n``.
        Возвращает список всех простых чисел ``p`` таких, что ``p <= n``.
        Граница ``n`` включается в результат, если она сама является простой.
        Для ``n < 2`` возвращает пустой список.

    Инвариант: после обработки очередного простого числа ``p`` все элементы
        ``is_prime[i]`` для ``i``, кратных ``p`` и ``i >= p^2``, установлены
        в ``False``. Все элементы ``is_prime[i] == True`` для ``i > 1``,
        не помеченные ранее, являются простыми.

    Сложность: O(n log log n) по времени, O(n) по памяти.

    Примеры:
        >>> primes(10)
        [2, 3, 5, 7]
        >>> primes(45)
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    """
    if n < 2:
        return []
    
    # Создаем логический массив (решето), где True означает, что число потенциально простое
    # Индексы от 0 до n, поэтому размер n + 1
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    # Проходим только до квадратного корня из n (используем math.isqrt для точности)
    for p in range(2, math.isqrt(n) + 1):
        if is_prime[p]:
            # Вычеркиваем кратные p, начиная с p^2
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
                
    # Собираем индексы, которые остались True, в список
    return [p for p in range(2, n + 1) if is_prime[p]]


def next_prime(n: int) -> int:
    """Возвращает наименьшее простое число p > n.

    Контракт:
        Принимает любое целое число ``n``.
        Возвращает целое простое число ``p``, строго большее ``n``.
        Для ``n < 2`` возвращает ``2``.

    Инвариант: на каждой итерации ``candidate`` является нечётным числом,
        большим ``n``, и все числа в диапазоне ``(n, candidate)``, которые
        проверялись ранее, не являются простыми.

    Raises:
        Ничего не возбуждает, так как множество простых чисел бесконечно
        (теорема Евклида), и ответ всегда существует.
    """
    if n < 2:
        return 2
    if n == 2:
        return 3
        
    # Если n нечетное, начинаем поиск с n + 2, иначе с n + 1
    # (так как n+1 будет четным, если n четное)
    candidate = n + 1
    if candidate % 2 == 0:
        candidate += 1
        
    # Перебираем только нечетные числа
    while not is_prime_fast(candidate):
        candidate += 2
        
    return candidate


def smallest_factor(n: int) -> int | None:
    """Возвращает наименьший простой делитель числа n >= 2.

    Контракт:
        Принимает целое число ``n``.
        Возвращает наименьший простой делитель числа ``n``. Если ``n`` —
        простое число, возвращает само ``n``.
        Для ``n < 2`` возвращает ``None``.

    Инвариант: на каждой итерации цикла текущее ``d`` проверяется на
        делимость. Так как перебор идет только по нечетным ``d`` (после
        обработки двойки), все составные делители пропускаются, и первым
        найденным делителем гарантированно будет простое число.

    Примеры:
        >>> smallest_factor(10)
        2
        >>> smallest_factor(9)
        3
        >>> smallest_factor(17)
        17
    """
    if n < 2:
        return None
    if n % 2 == 0:
        return 2
    for d in range(3, isqrt(n) + 1, 2):
        if n % d == 0:
            return d
    return n


def decompose(n: int) -> dict[int, int]:
    """Возвращает каноническое разложение числа n в виде словаря {p: e}.

    Контракт:
        Принимает целое число ``n >= 1``.
        Возвращает словарь, где ключи — простые делители числа ``n``,
        а значения — их степени (показатели) в разложении.
        Для ``n = 1`` возвращает пустой словарь.

    Инвариант: после каждой итерации ``n`` уменьшилось на найденный
        простой множитель ``p``, а счетчик ``D[p]`` увеличен на 1.
        Произведение ``p_i ** e_i`` для всех записанных пар в ``D``
        всегда равно исходному ``N``, деленному на текущее ``n``.

    Примеры:
        >>> decompose(12)
        {2: 2, 3: 1}
        >>> decompose(1)
        {}
    """
    original_n = n
    D = {}
    while n > 1:
        p = smallest_factor(n)
        D[p] = D.get(p, 0) + 1
        n //= p
    return D


def recompose(D: dict[int, int]) -> int:
    """Восстанавливает число по словарю канонического разложения {p: e}.

    Контракт:
        Принимает словарь ``D``, где ключи — простые числа ``p``,
        а значения — их степени ``e >= 1``.
        Возвращает целое число ``n >= 1``, равное произведению ``p**e``
        по всем парам в словаре.
        Для пустого словаря возвращает ``1`` (по определению, произведение
        по пустому множеству равно 1).

    Инвариант: после обработки ``k`` элементов словаря текущее значение
        ``n`` равно произведению ``p_i ** e_i`` для всех уже обработанных
        пар. В конце цикла ``n`` равно исходному числу.

    Примеры:
        >>> recompose({2: 3, 5: 3})
        1000
        >>> recompose({2: 1, 3: 1, 5: 1, 7: 1})
        210
        >>> recompose({})
        1
    """
    n = 1
    for p, e in D.items():
        n *= p ** e
    return n


def euler_phi(n: int) -> int:
    """Вычисляет функцию Эйлера phi(n) по разложению числа n.

    Контракт:
        Принимает целое число ``n >= 1``.
        Возвращает количество натуральных чисел ``k`` таких, что
        ``1 <= k <= n`` и ``gcd(k, n) = 1``.
        Для ``n = 1`` возвращает ``1`` (по определению).

    Формула: phi(n) = n * prod_{p|n} (1 - 1/p).
        Здесь произведение берется по всем простым делителям ``p`` числа ``n``.
        В целых числах это эквивалентно phi(n) = prod_{p^e || n} (p^e - p^(e-1)).

    Инвариант: после разложения ``n`` на простые множители ``{p: e}``
        результат вычисляется как произведение ``(p**e - p**(e-1))``
        по всем парам. Это значение всегда целое и не превосходит ``n``.

    Примеры:
        >>> euler_phi(9)
        6
        >>> euler_phi(1)
        1
        >>> euler_phi(1125)
        600
    """
    if n < 1:
        raise ValueError("n должно быть >= 1")
        
    # Для n = 1 функция возвращает 1
    if n == 1:
        return 1

    result = 1
    # Получаем разложение {p: e}
    for p, e in decompose(n).items():
        # Используем целочисленную формулу: p^e - p^(e-1)
        result *= (p ** e - p ** (e - 1))

    return result


def multiplicative_order(a: int, n: int, t: int | None = None) -> int:
    """Вычисляет порядок элемента a по модулю n.

    Контракт:
        Принимает целые числа ``a`` и ``n`` (``n >= 2``), а также
        необязательный параметр ``t`` — известное кратное порядка
        (по умолчанию ``t = phi(n)``).
        Возвращает наименьшее натуральное ``k >= 1``, такое что
        ``a^k ≡ 1 (mod n)``.
        Требует, чтобы ``gcd(a, n) = 1``.

    Инвариант: на каждом шаге ``t`` остается кратным истинному порядку.
        Если ``a^(t/q) ≡ 1 (mod n)``, то порядок делит ``t/q``, и мы
        можем безопасно уменьшить ``t``. В конце ни один простой
        делитель нельзя убрать, значит ``t = ord(a)``.

    Raises:
        ValueError: если ``a`` и ``n`` не взаимно просты.

    Примеры:
        >>> multiplicative_order(4, 9)
        3
        >>> multiplicative_order(2, 7)
        3
    """
    from math import gcd
    if gcd(a, n) != 1:
        raise ValueError(f"Элемент {a} не взаимно прост с {n}")

    if t is None:
        # t = euler_phi(n)
        t = 1
        for p, e in decompose(n).items():
            t *= p ** e - p ** (e - 1)

    # Ищем простые множители t
    factors = decompose(t).keys()

    # Для каждого простого q пробуем уменьшить t
    for q in factors:
        while t % q == 0 and mod_pow(a, t // q, n) == 1:
            t //= q

    return t


def is_primitive_root(g: int, n: int, t: int | None = None) -> bool:
    """Проверяет, является ли g первообразным корнем по модулю n.

    Контракт:
        Принимает целые числа ``g`` и ``n`` (``n >= 2``), а также
        необязательный параметр ``t`` — порядок группы (по умолчанию ``phi(n)``).
        Возвращает ``True``, если ``g`` — первообразный корень, иначе ``False``.
        Использует критерий (3.20): ``g^(phi(n)/q) != 1 (mod n)`` для
        каждого простого делителя ``q`` числа ``phi(n)``.

    Инвариант: если хотя бы для одного ``q`` условие ``g^(t/q) ≡ 1``
        выполняется, то порядок ``g`` меньше ``t``, и функция возвращает
        ``False``. Если условие не выполнено ни для одного ``q``,
        то порядок ``g`` равен ``t``, и функция возвращает ``True``.

    Примеры:
        >>> is_primitive_root(2, 9)
        True
        >>> is_primitive_root(4, 9)
        False
    """
    from math import gcd
    if gcd(g, n) != 1:
        return False

    if t is None:
        t = 1
        for p, e in decompose(n).items():
            t *= p ** e - p ** (e - 1)

    factors = decompose(t).keys()
    for q in factors:
        if mod_pow(g, t // q, n) == 1:
            return False
    return True


def primitive_root(n: int) -> int | None:
    """Находит наименьший первообразный корень по модулю n.

    Контракт:
        Принимает целое число ``n >= 2``.
        Возвращает наименьший первообразный корень ``g`` по модулю ``n``,
        или ``None``, если первообразных корней не существует (например,
        для ``n = 8``).

    Инвариант: перебирает ``g`` от 2 до ``n-1``. Как только находится
        ``g``, для которого ``is_primitive_root(g, n)`` возвращает ``True``,
        оно возвращается. Если цикл завершился без результата, возвращается ``None``.

    Примеры:
        >>> primitive_root(9)
        2
        >>> primitive_root(8) is None
        True
        >>> primitive_root(19)
        2
    """
    for g in range(2, n):
        if is_primitive_root(g, n):
            return g
    return None