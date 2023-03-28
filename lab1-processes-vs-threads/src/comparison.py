# %%
import multiprocessing as mp
from time import time
from typing import Callable
from typing import ClassVar
from multiprocessing import Process, Queue, Array, Value
from threading import Thread
from random import randrange
from sys import maxsize


def generate_list(size: int) -> list[int]:
    '''Generates a list of random numbers of length `size`, on numbers on `range(size)`'''
    return [randrange(size) for i in range(size)]
    # return [randrange(maxsize) for i in range(size)]


# %%
def merge(a: list[int], b: list[int]) -> list[int]:
    '''Merges two lists `a` and `b` into a single sorted list'''
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
    return result


def sort(list: list[int], job_id: int) -> list[int]:
    list.sort()
    # print(f"Job {job_id}:\tSorted {len(list)} elements")
    return list


# %%


def mergesort_thread(list: list[int], division: int) -> list[int]:
    '''Sorts a list using in-place non-recursive mergesort, where it splits the job size into `division` slices and sorts each slice in a single unit, then joing them back using units aswell'''
    if len(list) <= division:
        print("List is too small to split")
        list.sort()
        return list

    # Only the last element can be 1 smaller, only if necessary
    slices = []
    slice_size = len(list) // division
    for i in range(division):
        if i == division - 1:
            # Last slice does not need to be the same size
            slices.append(list[i * slice_size:])
        else:
            slices.append(list[i * slice_size:(i + 1) * slice_size])

    # Sort each slice in a job
    sort_jobs = [Thread(target=sort, args=(slice, i))
                 for (i, slice) in enumerate(slices)]
    for job in sort_jobs:
        job.start()

    # Sync point: join the divisions back
    for job in sort_jobs:
        job.join()

    # Merge pairs of the now sorted slices using jobs:
    while len(slices) > 1:
        merge_jobs = []
        for i in range(0, len(slices), 2):
            if i + 1 < len(slices):
                merge_jobs.append(
                    Thread(target=merge, args=(slices[i], slices[i + 1])))

        for job in merge_jobs:
            job.start()
        for job in merge_jobs:
            job.join()

        slices = [slices[i] for i in range(0, len(slices), 2)]

    return slices[0]


# %%


def sort_proc(slice, job_id: int):
    '''Sorts a list in a job, and puts it back in the queue'''
    list = [i for i in slice]
    # list = slice.tolist()
    list.sort()
    # print(f"No sort {list}")
    for i in range(len(list)):
        slice[i] = list[i]

    # print(f"Slice Depois do sort {slice}")
    return slice


def mergesort_process_shm(list: list[int], division: int) -> list[int]:
    '''Sorts a list using in-place non-recursive mergesort, where it splits the job size into `division` slices and sorts each slice in a single unit, then joing them back using units aswell'''
    if len(list) <= division:
        print("List is too small to split")
        list.sort()
        return list

    # list = Array('i', list)

    # Only the last element can be 1 smaller, only if necessary
    slices = []
    slice_size = len(list) // division
    for i in range(division):
        if i == division - 1:
            # Last slice does not need to be the same size
            slices.append(Array('i', list[i * slice_size:]))
        else:
            slices.append(
                Array('i', list[i * slice_size:(i + 1) * slice_size]))

    # Sort each slice in a job
    sort_jobs = [Process(target=sort_proc, args=(slice, i))
                 for (i, slice) in enumerate(slices)]
    for job in sort_jobs:
        job.start()

    # Sync point: join the divisions back
    for job in sort_jobs:
        job.join()

    # Merge pairs of the now sorted slices using jobs:
    while len(slices) > 1:
        merge_jobs = []
        for i in range(0, len(slices), 2):
            if i + 1 < len(slices):
                merge_jobs.append(
                    Process(target=merge, args=(slices[i], slices[i + 1])))
            else:
                # Only the last may not join, if the list is odd length
                merge_jobs.append(Process(target=merge, args=(slices[i], [])))

        for job in merge_jobs:
            job.start()
        for job in merge_jobs:
            job.join()

        slices = [slices[i] for i in range(0, len(slices), 2)]

    return slices[0]


# %%


def assert_is_ordered(list: list[int]):
    for i in range(len(list) - 1):
        assert list[i] <= list[i+1], f"list is not ordered: {list[i]} {list[i+1]}"


def run(list_size: int, rounds: int, parallels: int, func: Callable[list[int], int]) -> list[float]:
    '''gens 1 new list per round, to decrease variability'''

    times = []
    print(
        f"Running {func.__name__} with {parallels} units for list size {list_size} and {rounds} rounds")

    for round in range(rounds):
        list = generate_list(list_size)

        # record time elapsed
        start = time()
        list = func(list, parallels)
        elapsed = time() - start

        assert_is_ordered(list)
        # print(f"Round {round + 1}: \t{elapsed}")
        print(".", end="", flush=True)
        times.append(elapsed)
    print()

    return times


# %%

if __name__ == '__main__':
    list_sizes = [pow(10, i) for i in range(3)]
    parallels = [pow(2, i) for i in range(0, 5)]
    rounds = 100

    # run(list_size, rounds, units, func=mergesort_process_shm)
    import pandas as pd

    dfs = []
    for i, list_size in enumerate(list_sizes):
        for units in parallels:
            times = run(list_size, rounds, units, func=mergesort_thread)
            dfs.append(pd.DataFrame(
                times, columns=[f"{units}t\n{list_size}l"]))

    # for i, list_size in enumerate(list_sizes):
    #     for units in parallels:
    #         times = run(list_size, rounds, units, func=mergesort_process_shm)
    #         dfs.append(pd.DataFrame(
    #             times, columns=[f"{units}p\n{list_size}l"]))

    # %%

    df = pd.concat(dfs, axis=1)

    # %%
    import seaborn as sns
    import matplotlib as plt

    def plot(data: pd.DataFrame):
        '''Plots a list of lists of times, each list is a different algorithm'''
        # dict keys are labels and values are lists of times
        sns.set_theme()
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette("Set2")
        sns.boxplot(data=data, showmeans=True, meanprops={
                    "marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
        plt.pyplot.xlabel("Number of threads")
        plt.pyplot.ylabel("Time (s)")
        plt.pyplot.show()

    plot(df)
