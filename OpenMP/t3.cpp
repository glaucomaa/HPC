#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <chrono>
#include <iomanip>

const int MATRIX_SIZE = 1000;


// results
// Sequential Multiplication Result:
// Sequential multiplication time: 18.275107 seconds
// Parallel Multiplication Result:
// Parallel multiplication time: 4.940051 seconds


std::vector<std::vector<int>> generateMatrix(int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 10);

    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}

void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<int>> multiplySequential(const std::vector<std::vector<int>>& matrix1,
                                                 const std::vector<std::vector<int>>& matrix2) {
    int n = matrix1.size();

    std::vector<std::vector<int>> result(n, std::vector<int>(n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

std::vector<std::vector<int>> multiplyParallel(const std::vector<std::vector<int>>& matrix1,
                                               const std::vector<std::vector<int>>& matrix2) {
    int n = matrix1.size();

    std::vector<std::vector<int>> result(n, std::vector<int>(n));

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
#pragma omp atomic
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

int main() {
    std::vector<std::vector<int>> matrix1 = generateMatrix(MATRIX_SIZE);
    std::vector<std::vector<int>> matrix2 = generateMatrix(MATRIX_SIZE);

    // std::cout << "Matrix 1:" << std::endl;
    // printMatrix(matrix1);

    // std::cout << "Matrix 2:" << std::endl;
    // printMatrix(matrix2);

    std::cout << "Sequential Multiplication Result:" << std::endl;
    auto startSeq = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> resultSequential = multiplySequential(matrix1, matrix2);
    auto endSeq = std::chrono::steady_clock::now();
    std::chrono::duration<double> durationSeq = endSeq - startSeq;
    // printMatrix(resultSequential);
    std::cout << "Sequential multiplication time: " << std::fixed << std::setprecision(6) << durationSeq.count() << " seconds" << std::endl;
    
    std::cout << "Parallel Multiplication Result:" << std::endl;
    auto startParallel = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> resultParallel = multiplyParallel(matrix1, matrix2);
    auto endParallel = std::chrono::steady_clock::now();
    std::chrono::duration<double> durationParallel = endParallel - startParallel;
    // printMatrix(resultParallel);
    std::cout << "Parallel multiplication time: " << std::fixed << std::setprecision(6) << durationParallel.count() << " seconds" << std::endl;

    return 0;
}
