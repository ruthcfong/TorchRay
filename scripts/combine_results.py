import argparse
import os

import numpy as np


def combine_results(results_dir, out_path):
    results = []
    for f in os.listdir(results_dir):
        results.append(np.loadtxt(os.path.join(results_dir, f), dtype=str))
    results = np.array(results)
    np.savetxt(out_path, results, fmt='%s', delimiter='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',
                        type=str,
                        default='./data/attribution_benchmarks')
    parser.add_argument('--out_path',
                        type=str,
                        default='./data/attribution_benchmarks_all.csv')

    args = parser.parse_args()

    combine_results(args.results_dir, args.out_path)
