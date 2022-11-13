#!/usr/bin/env sage
import numpy as np

def solveb(mat):
    
    # nrow, ncol, readmatrix = eval(input("Please input:\nnrow, ncol, and the matrix\n"))
    nrow, ncol = mat.shape
    
    a = matrix(
        GF(2),
        nrow,
        ncol,
        mat
    )
    
    b = a.right_kernel()
    
    return b

if __name__ == "__main__":
    mat = np.loadtxt("matrix.csv", dtype=int)
    # print(mat)

    filename = "vector-b.csv"

    b = solveb(mat)

    s = str(b.matrix())[1:-1]

    # print(b.matrix())

    with open(filename, "w") as f:
        f.write(s)

    print(f"The following result saved to {filename}.")

    print(b)
