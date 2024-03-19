import numpy as np


#method 1 convemntional approach
def matrix_multiply(matrix1, matrix2):
    # Check if matrices are compatible for multiplication
    if len(matrix1[0]) != len(matrix2):
        print("Matrices are not compatible for multiplication.")
        return None
    else:
        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix2[0])):
                sum = 0
                for k in range(len(matrix2)):
                    sum += matrix1[i][k] * matrix2[k][j]
                row.append(sum)
            result.append(row)
        return result
    

#Method 2 Divide and Conquer
def split_matrix(matrix):
    """
    Split a given matrix into four quadrants.
    """
    rows, cols = matrix.shape
    half_rows, half_cols = rows // 2, cols // 2
    A11 = matrix[:half_rows, :half_cols]
    A12 = matrix[:half_rows, half_cols:]
    A21 = matrix[half_rows:, :half_cols]
    A22 = matrix[half_rows:, half_cols:]
    return A11, A12, A21, A22

def combine_matrices(C11, C12, C21, C22):
    """
    Combine the four quadrants to form a single matrix.
    """
    top_half = np.concatenate((C11, C12), axis=1)
    bottom_half = np.concatenate((C21, C22), axis=1)
    return np.concatenate((top_half, bottom_half), axis=0)

def divide_and_conquer_matrix_multiply(A, B):
    """
    Divide and conquer algorithm for matrix multiplication.
    """
    if min(A.shape) <= 2:  # Base case
        return np.dot(A, B)
    
    # Split matrices into sub-matrices
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    # Recursively compute sub-matrix multiplications
    C11 = divide_and_conquer_matrix_multiply(A11, B11) + divide_and_conquer_matrix_multiply(A12, B21)
    C12 = divide_and_conquer_matrix_multiply(A11, B12) + divide_and_conquer_matrix_multiply(A12, B22)
    C21 = divide_and_conquer_matrix_multiply(A21, B11) + divide_and_conquer_matrix_multiply(A22, B21)
    C22 = divide_and_conquer_matrix_multiply(A21, B12) + divide_and_conquer_matrix_multiply(A22, B22)
    
    # Combine sub-matrix results to obtain the final product matrix
    return combine_matrices(C11, C12, C21, C22)


#Method 3 Strassen's approach
def strassen_algorithm(x, y):
    if x.size == 1 or y.size == 1:
        return x * y

    n = x.shape[0]

    if n % 2 == 1:
        x = np.pad(x, (0, 1), mode='constant')
        y = np.pad(y, (0, 1), mode='constant')

    m = int(np.ceil(n / 2))
    a = x[: m, : m]
    b = x[: m, m:]
    c = x[m:, : m]
    d = x[m:, m:]
    e = y[: m, : m]
    f = y[: m, m:]
    g = y[m:, : m]
    h = y[m:, m:]
    p1 = strassen_algorithm(a, f - h)
    p2 = strassen_algorithm(a + b, h)
    p3 = strassen_algorithm(c + d, e)
    p4 = strassen_algorithm(d, g - e)
    p5 = strassen_algorithm(a + d, e + h)
    p6 = strassen_algorithm(b - d, g + h)
    p7 = strassen_algorithm(a - c, e + f)
    result = np.zeros((2 * m, 2 * m), dtype=np.int32)
    result[: m, : m] = p5 + p4 - p2 + p6
    result[: m, m:] = p1 + p2
    result[m:, : m] = p3 + p4
    result[m:, m:] = p1 + p5 - p3 - p7

    return result[: n, : n]

        
if __name__ == "__main__":
    Matrix1 = np.array([[1, 2, 3, 4, 5, 6,3],
           [4, 5, 6, 4, 5, 6,3],
           [7, 8, 9, 4, 5, 6,3],
           [1, 2, 3, 4, 5, 6,3],
           [1, 2, 3, 4, 5, 6,3],
           [6, 5, 4, 3, 2, 1,3]])
    Matrix2 = np.array([[1, 2, 3, 4, 5, 6,3],
           [4, 5, 6, 4, 5, 6,3],
           [7, 8, 9, 4, 5, 6,3],
           [1, 2, 3, 4, 5, 6,3],
           [1, 2, 3, 4, 5, 6,3],
           [6, 5, 4, 3, 2, 1,3]])
    

    C_result = matrix_multiply(Matrix1, Matrix2)
    if C_result:  #if result's flag is true then this will print the resultant matrix
        print("Result of matrix multiplication using Conventional Approach:")
        for row in C_result:
            print(row)
    

    D_C_result = divide_and_conquer_matrix_multiply(Matrix1, Matrix2)
    print("Result of matrix multiplication using Divide and Conquer Approach:")
    print(D_C_result)
    
    print('Matrix multiplication result using Strassens Approach: ')
    print(strassen_algorithm(Matrix1, Matrix2))




