import numpy as np
from tutorial_utils import print_details

# arange returns arrays instead of lists
a = np.arange(15).reshape(3, 5)

# zeros fill array with 0 points
b = np.zeros(shape=(3, 10))

np.array([
       [0,  1,  2,  3,  4],
       [5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]
], dtype=np.int32)

print_details(a)
print_details(b)

