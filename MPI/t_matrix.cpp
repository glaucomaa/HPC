#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

void generateMatrix(int** matrix, int N) {
    srand(time(0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() % 10;
        }
    }
}

void printMatrix(int** matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << setw(4) << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

void multiplySequential(int** matrixA, int** matrixB, int** result, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0;
            for (int k = 0; k < N; k++) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

void multiplyParallel(int** matrixA, int** matrixB, int** result, int N, int numProcesses, int rank) {
    int chunkSize = N / numProcesses;
    int startRow = rank * chunkSize;
    int endRow = startRow + chunkSize;

    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0;
            for (int k = 0; k < N; k++) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    MPI_Allgather(MPI_IN_PLACE, chunkSize * N, MPI_INT, result[0], chunkSize * N, MPI_INT, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int sizes[] = {5, 10, 50, 100, 500, 1000};
    int numProcesses[] = {1, 2, 4, 8, 16, 32, 64, 128};

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int size : sizes) {
        int** matrixA = new int*[size];
        int** matrixB = new int*[size];
        int** resultSequential = new int*[size];
        int** resultParallel = new int*[size];
        for (int i = 0; i < size; i++) {
            matrixA[i] = new int[size];
            matrixB[i] = new int[size];
            resultSequential[i] = new int[size];
            resultParallel[i] = new int[size];
        }

        generateMatrix(matrixA, size);
        generateMatrix(matrixB, size);

        double startTimeSequential = MPI_Wtime();
        multiplySequential(matrixA, matrixB, resultSequential, size);
        double endTimeSequential = MPI_Wtime();

        for (int numProc : numProcesses) {
            MPI_Barrier(MPI_COMM_WORLD);
            double startTimeParallel = MPI_Wtime();

            MPI_Comm newComm;
            MPI_Comm_split(MPI_COMM_WORLD, rank < numProc, rank, &newComm);

            int newRank;
            MPI_Comm_rank(newComm, &newRank);

            if (newRank != MPI_UNDEFINED) {
                multiplyParallel(matrixA, matrixB, resultParallel, size, numProc, newRank);
            }

            double endTimeParallel = MPI_Wtime();

            if (rank == 0) {
                cout << "Size: " << size << " x " << size << " | Processes: " << numProc << " | ";

                bool correct = true;
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        if (resultSequential[i][j] != resultParallel[i][j]) {
                            correct = false;
                            break;
                        }
                    }
                    if (!correct) {
                        break;
                    }
                }

                if (correct) {
                    // cout << "Result is correct | ";
                } else {
                    // cout << "Result is incorrect | ";
                }

                cout << "Sequential Execution Time: " << endTimeSequential - startTimeSequential << " seconds | ";
                cout << "Parallel Execution Time: " << endTimeParallel - startTimeParallel << " seconds" << endl;
            }

            MPI_Comm_free(&newComm);
        }

        for (int i = 0; i < size; i++) {
            delete[] matrixA[i];
            delete[] matrixB[i];
            delete[] resultSequential[i];
            delete[] resultParallel[i];
        }
        delete[] matrixA;
        delete[] matrixB;
        delete[] resultSequential;
        delete[] resultParallel;
    }

    MPI_Finalize();

    return 0;
}
