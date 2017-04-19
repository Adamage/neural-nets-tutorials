def print_details(a):
    print("Shape: " + str(a.shape))
    print("Ndim: " + str(a.ndim))
    print("Dtype.name: " + str(a.dtype.name))
    print("Size: " + str(a.size))
    print("Type: " + str(type(a)))
    print(a)
    print(3 * "\n")


def insert_print_space():
    print(3 * "\n")