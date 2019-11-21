import argparse

import matplotlib.pyplot as plt
import numpy as np


def plot(path_1, path_2, xlabel="", ylabel="", title=""):
    x = np.loadtxt(path_1)
    y = np.loadtxt(path_2)

    f, ax = plt.subplots(1, 1)
    ax.plot(x, y, '.')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_1")
    parser.add_argument("path_2")
    parser.add_argument("--xlabel", default="")
    parser.add_argument("--ylabel", default="")
    parser.add_argument("--title", default="")

    args = parser.parse_args()

    plot(args.path_1,
         args.path_2,
         xlabel=args.xlabel,
         ylabel=args.ylabel,
         title=args.title)
