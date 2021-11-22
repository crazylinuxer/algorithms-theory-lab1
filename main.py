import time
import json
import subprocess
from typing import List, Union, Tuple
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


def get_abc(arr_n: List[int], arr_t: List[Union[int, float]]) -> Tuple[float, float, float]:
    sum_n = sum(arr_n)
    sum_n2 = sum(i ** 2 for i in arr_n)
    sum_t = sum(arr_t)

    sum_tn = sum(n * t for n, t in zip(arr_n, arr_t))
    sum_tnlogn = sum(n * t * numpy.log2(n) for n, t in zip(arr_n, arr_t))
    sum_nlogn = sum(n * numpy.log2(n) for n in arr_n)
    sum_n2logn = sum((n ** 2) * numpy.log2(n) for n in arr_n)
    sum_n2logn2 = sum((n ** 2) * numpy.log2(n) ** 2 for n in arr_n)

    system = numpy.array([
        [sum_n2logn2, sum_n2logn, sum_nlogn],
        [sum_n2logn, sum_n2, sum_n],
        [sum_nlogn, sum_n, len(arr_n)]
    ])
    t_1 = numpy.array([sum_tnlogn, sum_tn, sum_t])
    return tuple(numpy.linalg.solve(system, t_1))


def main():
    def clear():
        print("\r", end='')
        print(" " * 8, end='')
        print("\r", end='')

    def nlogn(a_coefficient, b_coefficient, c_coefficient, n):
        return a_coefficient * n * numpy.log2(n) + b_coefficient * n + c_coefficient

    # array_sizes = [*(i * 100 for i in range(1, 200, 2)), *(i * 20000 for i in range(1, 1001))]
    array_sizes = [i * 10 for i in range(1, 1000)]
    time_stats = []
    last_print_time = 0
    executor = ThreadPoolExecutor(max_workers=8)
    iterator = executor.map(lambda x: time_stats.append(get_sorting_time(x, 8)), array_sizes)
    counter = 0
    print("0.0%", end='')
    for _ in iterator:
        if (time.time() - last_print_time) > 0.5:
            clear()
            print(round((counter / len(array_sizes)) * 100, 2), end='%', flush=True)
            last_print_time = time.time()
        counter += 1

    executor.shutdown()
    clear()
    a, b, c = get_abc(array_sizes, time_stats)
    print(
        f"{a:.12f}*n*log2(n) {'+' if b >= 0 else '-'} {numpy.abs(b):.12f}*n {'+' if c >= 0 else '-'} {numpy.abs(c):.12f} = T"
    )
    theoretical_time_stats = [nlogn(a, b, c, n) for n in array_sizes]
    json_data = {
        "practical": {
            str(a): b for a, b in zip(array_sizes, time_stats)
        },
        "theoretical": {
            str(a): b for a, b in zip(array_sizes, theoretical_time_stats)
        }
    }
    with open("results.json", "w") as out:
        json.dump(json_data, out, indent=4)
    plt.figure(figsize=(24, 12))
    plt.plot(array_sizes, time_stats, linewidth=0.7)
    plt.plot(array_sizes, theoretical_time_stats, color="orange")
    plt.savefig("./graph_pictures/auto.png", dpi=400)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExit")
