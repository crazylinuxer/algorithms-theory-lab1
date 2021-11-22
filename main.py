import time
import subprocess
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor

from matplotlib import pyplot as plt
import numpy


def get_sorting_time(count: int, tests: int) -> float:
    raw_result = subprocess.run(
        ["./inplace_heap_sort/inplace_heap_sort", str(count), str(tests)],
        capture_output=True
    )
    if raw_result.returncode != 0:
        raise RuntimeError(
            f"Error in sorting process! Code: {raw_result.returncode}. Stderr: '{raw_result.stderr.decode()}'"
        )
    return float(raw_result.stdout.decode().split('\n')[-2].split()[1])


def get_abc(arr_n: List[int], arr_t: List[Union[int, float]]) -> List[float]:
    sum_n = sum(arr_n)
    sum_n2 = sum(i ** 2 for i in arr_n)
    sum_t = sum(arr_t)

    sum_tn = sum(n * t for n, t in zip(arr_n, arr_t))
    sum_tnlogn = sum(n * t * numpy.log2(n) for n, t in zip(arr_n, arr_t))
    sum_nlogn = sum(n * numpy.log2(n) for n in arr_n)
    sum_n2logn = sum((n ** 2) * numpy.log2(n) for n in arr_n)
    sum_n2logn2 = sum((n ** 2) * numpy.log2(n) ** 2 for n in arr_n)

    system_1 = numpy.array([
        [sum_n2logn2, sum_n2logn, sum_nlogn],
        [sum_n2logn, sum_n2, sum_n],
        [sum_nlogn, sum_n, len(arr_n)]
    ])
    t_1 = numpy.array([sum_tnlogn, sum_tn, sum_t])
    return numpy.linalg.solve(system_1, t_1)


def main():
    def clear():
        print("\r", end='')
        print(" " * 8, end='')
        print("\r", end='')

    def nlogn(a, b, c, n):
        return a * n * numpy.log2(n) + b * n + c

    array_sizes = [i * 2000 for i in range(1, 20004, 4)]
    # array_sizes = [1, 2, 4, 8, *(i * 10 for i in range(1, 20))]
    time_stats = []
    last_print_time = 0
    executor = ThreadPoolExecutor(max_workers=8)
    iterator = executor.map(lambda x: time_stats.append(get_sorting_time(x, 16)), array_sizes)
    counter = 0
    print("0.0%", end='')
    for _ in iterator:
        if (time.time() - last_print_time) > 0.2:
            clear()
            print(round((counter / len(array_sizes)) * 100, 2), end='%', flush=True)
            last_print_time = time.time()
        counter += 1

    executor.shutdown()
    clear()
    print()
    abc = get_abc(array_sizes, time_stats)
    theoretical_time_stats = [nlogn(*abc, n) for n in array_sizes]
    plt.plot(array_sizes, time_stats)
    plt.plot(array_sizes, theoretical_time_stats, color="orange")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExit")
