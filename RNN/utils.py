import io
import os
import unicodedata
import string
import glob

import torch
import random

# Alphabet small + capital letters + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


# Turn a Unicode string to plain ASCII
def unicode_to_ascii(s) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_LETTERS
    )


def load_data() -> tuple:
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)

    # Read a file and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding="utf-8").read().strip().split("\n")
        return [unicode_to_ascii(line) for line in lines]

    for filename in find_files("../data/names/*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories


"""
To represent a single letter, we use a "one-hot vector" of size <1 x n_letters>.
A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.

To make a word we join a bunch of those into a 2D matrix <line_length x 1 x n_letters>.

That extra 1 dimension is because PyTorch assumes everything is in batches - we're just using a batch size of 1 here.
"""


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter) -> int:
    return ALL_LETTERS.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter) -> torch.Tensor:
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters> tensor
def line_to_tensor(line) -> torch.Tensor:
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


# Get a random training example
def random_training_example(category_lines, all_categories):
    def random_choice(a):
        return a[random.randint(0, len(a) - 1)]

    def random_training_pair():
        category = random_choice(all_categories)
        line = random_choice(category_lines[category])
        category_tensor = torch.tensor(
            [all_categories.index(category)], dtype=torch.long
        )
        line_tensor = line_to_tensor(line)
        return category, line, category_tensor, line_tensor

    return random_training_pair()
