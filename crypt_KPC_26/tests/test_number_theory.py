import pytest
import math
import random
from crypto_math.number_theory import (
    is_prime_fast, primes, next_prime,
    recompose, decompose, euler_phi, 
    multiplicative_order, mod_pow
)

def test_is_prime_fast_and_primes_consistency():
    """Проверяет, что is_prime_fast и primes согласованы на 0 <= n <= 10^4."""
    limit = 10_000
    
    # Получаем эталонный список простых чисел с помощью решета
    prime_list = primes(limit)
    prime_set = set(prime_list)
    
    for n in range(limit + 1):
        # Проверяем, что результат is_prime_fast совпадает с наличием числа в списке простых
        assert is_prime_fast(n) == (n in prime_set), \
            f"Расхождение для n={n}: is_prime_fast={is_prime_fast(n)}, primes содержит={n in prime_set}"


def test_primes_list_properties():
    """Проверяет свойства списка, возвращаемого функцией primes()."""
    limit = 500
    result = primes(limit)
    
    # 1. Список отсортирован по возрастанию
    assert result == sorted(result)
    
    # 2. Все числа в списке действительно простые
    for p in result:
        assert is_prime_fast(p), f"Число {p} в списке не является простым"
        
    # 3. В списке нет дубликатов
    assert len(result) == len(set(result))
    
    # 4. Все простые числа до limit включены (проверяем выборочно)
    for n in range(2, limit + 1):
        if is_prime_fast(n):
            assert n in result, f"Простое число {n} пропущено"


def test_next_prime_properties():
    """Проверяет свойства функции next_prime(n)."""
    limit = 1000
    
    for n in range(limit + 1):
        p = next_prime(n)
        
        # 1. Результат должен быть простым числом
        assert is_prime_fast(p), f"next_prime({n}) = {p} не является простым"
        
        # 2. Результат должен быть строго больше n
        assert p > n, f"next_prime({n}) = {p} не больше {n}"
        
        # 3. Между n и p не должно быть простых чисел
        # (проверяем, что нет простых x таких, что n < x < p)
        for x in range(n + 1, p):
            assert not is_prime_fast(x), \
                f"Между n={n} и next_prime={p} найдено простое число {x}"


def test_edge_cases():
    """Проверяет граничные случаи для всех функций."""
    # Для is_prime_fast
    assert is_prime_fast(-5) is False
    assert is_prime_fast(0) is False
    assert is_prime_fast(1) is False
    assert is_prime_fast(2) is True
    
    # Для primes
    assert primes(-10) == []
    assert primes(0) == []
    assert primes(1) == []
    assert primes(2) == [2]
    
    # Для next_prime
    assert next_prime(-100) == 2
    assert next_prime(0) == 2
    assert next_prime(1) == 2
    assert next_prime(2) == 3


def test_recompose_decompose_identity():
    """Свойство 1: recompose(decompose(n)) == n для всех n от 1 до 10000."""
    for n in range(1, 10_001):
        d = decompose(n)
        reconstructed = recompose(d)
        assert reconstructed == n, f"Ошибка для n={n}: {d} -> {reconstructed}"


def test_euler_theorem():
    """Свойство 2: a^phi(n) == 1 (mod n) для случайных обратимых a."""
    test_moduli = [10, 25, 100, 512, 1000]
    
    for n in test_moduli:
        phi_n = euler_phi(n)
        # Находим все обратимые элементы
        invertibles = [a for a in range(1, n) if math.gcd(a, n) == 1]
        
        # Берем 10 случайных (если элементов много) или все
        sample_size = min(10, len(invertibles))
        samples = random.sample(invertibles, sample_size)
        
        for a in samples:
            result = mod_pow(a, phi_n, n)
            assert result == 1, f"Ошибка: {a}^{phi_n} mod {n} = {result} (должно быть 1)"


def test_multiplicative_order_divides_phi():
    """Свойство 3: multiplicative_order(a, n) делит euler_phi(n)."""
    test_moduli = [9, 15, 27, 34, 101]
    
    for n in test_moduli:
        phi_n = euler_phi(n)
        for a in range(1, n):
            if math.gcd(a, n) == 1:
                order = multiplicative_order(a, n)
                assert phi_n % order == 0, \
                    f"Ошибка: ord({a})={order} не делит phi({n})={phi_n}"


def test_sum_of_phi_over_divisors():
    """Свойство 4: сумма phi(d) по всем делителям d числа n равна n."""
    for n in range(1, 1001):
        # Находим все делители n
        divisors = [d for d in range(1, n + 1) if n % d == 0]
        
        # Суммируем функцию Эйлера для каждого делителя
        sum_phi = sum(euler_phi(d) for d in divisors)
        
        assert sum_phi == n, f"Ошибка для n={n}: сумма={sum_phi}, ожидалось {n}"