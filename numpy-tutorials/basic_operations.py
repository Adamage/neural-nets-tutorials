import numpy as np
from tutorial_utils import insert_print_space


# Arithmetic operators on arrays apply elementwise. A new array is created and filled with the result.

A = np.array([[1,1], [0,2]])
B = np.array([[2,0], [3,4]])

# elementwise PRODUCT
print(A*B)
insert_print_space()

# elementwise MATRIX "DOT" PRODUCT
print(A.dot(B))
insert_print_space()
print(np.dot(A, B))
insert_print_space()
