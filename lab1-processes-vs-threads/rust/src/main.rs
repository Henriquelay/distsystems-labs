use std::{
    thread,
    time::{Duration, Instant},
};

/// Sorts the list in place and returns a reference to the list.
fn sort<T: Ord>(list: &mut [T]) -> &[T] {
    list.sort();
    list
}

/// Merges lists of sorted lists into a single sorted list
fn merge<T: PartialOrd>(
    list1: impl IntoIterator<Item = T>,
    list2: impl IntoIterator<Item = T>,
) -> Vec<T> {
    let mut list1 = list1.into_iter();
    let mut list2 = list2.into_iter();
    let mut res = Vec::new();

    loop {
        let a = list1.next();
        let b = list2.next();
        match (a, b) {
            (None, None) => break,
            (None, Some(v)) | (Some(v), None) => res.push(v),
            (Some(a), Some(b)) => {
                if a < b {
                    res.push(a);
                    res.push(b);
                } else {
                    res.push(b);
                    res.push(a);
                }
            }
        }
    }
    res
}

// dispatch `threads` threads running `sort` on `list`, returning list of handles to the threads
fn job_thread(list: &[usize], threads: usize) -> Vec<usize> {
    let handles = {
        let chunk_size = list.len() / threads;

        let mut handles = Vec::new();
        let starting_index = 0;
        let end_index = list.len();
        for i in (starting_index..end_index).step_by(chunk_size) {
            let end = i + chunk_size;
            let mut list = list.clone();
            let handle = thread::spawn(move || sort(&mut list[i..end]).to_vec());
            handles.push(handle);
        }

        handles
    };

    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();

    // merges the results of the threads using the `merge` function, dispatching a thread for each pair of results
    let results = {
        let mut handles = Vec::new();
        let mut i = 0;
        while i < results.len() {
            let list1 = results[i].clone();
            let list2 = results[i + 1].clone();
            let handle = thread::spawn(move || merge(list1, list2));
            handles.push(handle);
            i += 2;
        }
        results
    };

    results[0].to_vec()
}

fn gen_random_list(list_size: usize) -> Vec<usize> {
    let mut list = Vec::new();
    for _ in 0..list_size {
        list.push(rand::random::<usize>());
    }
    list
}

fn assert_is_sorted(list: &[usize]) {
    for i in 0..list.len() - 1 {
        assert!(list[i] <= list[i + 1]);
    }
}

fn diagnostics(list: Vec<usize>, units: usize, rounds: usize) -> Vec<Duration> {
    let mut results = Vec::new();
    for round in 0..rounds {
        let now = Instant::now();
        let sorted_list = job_thread(&list, units);
        let elapsed = now.elapsed();
        assert_is_sorted(&sorted_list);
        print!(".");
        results.push(elapsed);
    }
    results
}

fn main() {
    let list_sizes = [10_000, 100_000, 1_000_000, 10_000_000];
    let threads = [1, 2, 4, 8, 16, 32, 64, 128, 256];
    let rounds = 10;

    let threads_results = {
        println!("Running threads");
        let mut threads_results = vec![];
        for &list_size in &list_sizes {
            println!("List size: {}", list_size);
            let list = gen_random_list(list_size);
            let mut list_results = vec![];
            for &units in &threads {
                println!("Threads: {}", units);
                let unit_results = diagnostics(list, units, rounds);
                list_results.push(unit_results);
            }
            threads_results.push(list_results);
        }
        threads_results
    };
}
