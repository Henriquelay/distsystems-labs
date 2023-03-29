# %%


def generate_list(size: int) -> list[int]:
    """Generates a list of random numbers of length `size`, on numbers on `range(size)`"""

    from random import randrange
    from sys import maxsize

    return [randrange(size) for i in range(size)]
    # return [randrange(maxsize) for i in range(size)]


# %%


from queue import Queue
from typing import Type


def merge(a: list[int], b: list[int], q: Type[Queue]) -> list[int]:
    """Merges two lists `a` and `b` into a single sorted list"""

    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result += a[i:]
    result += b[j:]
    q.put(result)


def sort(list: list[int], job_id: int) -> list[int]:
    list.sort()


# %%


def mergesort_thread(list: list[int], division: int) -> list[int]:
    """Sorts a list using in-place non-recursive mergesort, where it splits the job size into `division` slices and sorts each slice in a single unit, then joing them back using units aswell"""

    from threading import Thread

    if len(list) <= division:
        # print("List is too small to split")
        list.sort()
        return list

    # Only the last element can be 1 smaller, only if necessary
    slices = []
    slice_size = len(list) // division
    for i in range(division):
        if i == division - 1:
            # Last slice does not need to be the same size
            slices.append(list[i * slice_size :])
        else:
            slices.append(list[i * slice_size : (i + 1) * slice_size])

    # Sort each slice in a job
    sort_jobs = [
        Thread(target=sort, args=(slice, i)) for (i, slice) in enumerate(slices)
    ]
    for job in sort_jobs:
        job.start()

    # Sync point: join the divisions back
    for job in sort_jobs:
        job.join()

    q = Queue()

    # Merge pairs of the now sorted slices using jobs:
    while len(slices) > 1:
        merge_jobs = []
        for i in range(0, len(slices), 2):
            if i + 1 < len(slices):
                merge_jobs.append(
                    Thread(target=merge, args=(slices[i], slices[i + 1], q))
                )

        for job in merge_jobs:
            job.start()
        for job in merge_jobs:
            job.join()

        new_slices = [q.get() for _ in range(len(merge_jobs))]
        if len(slices) % 2 == 1:
            # Only the last may not join, if the list is odd length
            new_slices.append(slices[-1])
        slices = new_slices

    return slices[0]


# %%

from multiprocessing import Queue


def sort_proc(slice: list[int], job_id: int, q: Type[Queue]):
    """Sorts a list in a job, and puts it back in the queue"""

    sort(slice, job_id)
    q.put(slice)


def merge_proc(a: list[int], b: list[int], q: Type[Queue]) -> list[int]:
    """Merges two lists `a` and `b` into a single sorted list, and puts it back in the queue"""

    merge(a, b, q)


def mergesort_process_shm(list: list[int], division: int) -> list[int]:
    """Sorts a list using in-place non-recursive mergesort, where it splits the job size into `division` slices and sorts each slice in a single unit, then joing them back using units aswell"""

    from multiprocessing import Process

    if len(list) <= division:
        # print("List is too small to split")
        list.sort()
        return list

    # Only the last element can be 1 smaller, only if necessary
    slices = []
    slice_size = len(list) // division
    for i in range(division):
        if i == division - 1:
            # Last slice does not need to be the same size
            slices.append(list[i * slice_size :])
        else:
            slices.append(list[i * slice_size : (i + 1) * slice_size])

    q = Queue()
    # Sort each slice in a job
    sort_jobs = [
        Process(target=sort_proc, args=(slice, i, q))
        for (i, slice) in enumerate(slices)
    ]

    for job in sort_jobs:
        job.start()

    # Sync point: join the divisions back
    for job in sort_jobs:
        job.join()

    slices = [q.get() for _ in sort_jobs]

    assert q.empty()

    # Merge pairs of the now sorted slices using jobs:
    while len(slices) > 1:
        merge_jobs = []
        for i in range(0, len(slices), 2):
            if i + 1 < len(slices):
                merge_jobs.append(
                    Process(target=merge_proc, args=(slices[i], slices[i + 1], q))
                )
            # else:
            #     # Only the last may not join, if the list is odd length
            #     merge_jobs.append(Process(target=merge, args=(slices[i], [])))

        for job in merge_jobs:
            job.start()
        for job in merge_jobs:
            job.join()

        new_slices = [q.get() for _ in range(len(merge_jobs))]
        if len(slices) % 2 == 1:
            # Only the last may not join, if the list is odd length
            new_slices.append(slices[-1])
        slices = new_slices

    return slices[0]


# %%


def assert_is_ordered(slice: list[int], list_size):
    # print(f"Ordenado?: {list}")
    assert len(slice) == list_size, "Elements are missing"
    for i in range(len(slice) - 1):
        assert (
            slice[i] <= slice[i + 1]
        ), f"slice is not ordered: {slice[i]} {slice[i+1]}"


from typing import Callable


def run(
    list_size: int,
    rounds: int,
    parallels: int,
    func: Callable[[list[int], int], list[int]],
) -> list[float]:
    """gens 1 new list per round, to decrease variability"""

    from time import time

    times = []
    print(
        f"{func.__name__}\t|\t{parallels} units\t|\tsize {list_size}\t|\t{rounds} rounds"
    )

    for round in range(rounds):
        slice = generate_list(list_size)
        assert len(slice) == list_size, "wrong list size"

        # record time elapsed
        start = time()
        slice = func(slice, parallels)
        elapsed = time() - start

        assert_is_ordered(slice, list_size)
        # print(f"Round {round + 1}: \t{elapsed}")
        print(".", end="", flush=True)
        times.append(elapsed)
    print()

    return times


# %%

if __name__ == "__main__":
    list_sizes = [pow(10, i) for i in range(3)]
    parallels = [pow(2, i) for i in range(0, 5)]
    rounds = 10

    run(10, rounds, 2, func=mergesort_thread)
    import pandas as pd

    dfs = []
    for i, list_size in enumerate(list_sizes):
        for units in parallels:
            times = run(list_size, rounds, units, func=mergesort_thread)
            dfs.append(pd.DataFrame(times, columns=[f"{units}t\n{list_size}l"]))

    for i, list_size in enumerate(list_sizes):
        for units in parallels:
            times = run(list_size, rounds, units, func=mergesort_process_shm)
            dfs.append(pd.DataFrame(times, columns=[f"{units}p\n{list_size}l"]))

    # %%

    df = pd.concat(dfs, axis=1)

    # %%

    def plot(data: pd.DataFrame):
        """Plots a list of lists of times, each list is a different algorithm"""
        import seaborn as sns
        import matplotlib as plt

        # dict keys are labels and values are lists of times
        sns.set_theme()
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette("Set2")
        sns.boxplot(
            data=data,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
            },
        )
        plt.pyplot.xlabel("Number of threads")
        plt.pyplot.ylabel("Time (s)")
        plt.pyplot.show()

    plot(df)
