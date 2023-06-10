#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <mpi.h>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1);
    std::vector<int> R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            ++i;
        }
        else {
            arr[k] = R[j];
            ++j;
        }
        ++k;
    }

    while (i < n1) {
        arr[k] = L[i];
        ++i;
        ++k;
    }

    while (j < n2) {
        arr[k] = R[j];
        ++j;
        ++k;
    }
}

void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int main(int argc, char** argv) {
    int size = 1000000;
    int rangeMin = 0;    
    int rangeMax = 1000000;

    int maxProcesses = 128;

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int numProcesses = 1; numProcesses <= maxProcesses; numProcesses *= 2) {
        int subSize = size / numProcesses;
        std::vector<int> subArr(subSize);

        if (rank == 0) {
            std::vector<int> arr(size);

            srand(static_cast<unsigned>(time(0)));
            for (int i = 0; i < size; ++i)
                arr[i] = rand() % (rangeMax - rangeMin + 1) + rangeMin;


            // std::cout << "Unsorted Array: ";
            // for (int i = 0; i < size; ++i)
                // std::cout << arr[i] << " ";
            // std::cout << std::endl;

            MPI_Scatter(arr.data(), subSize, MPI_INT, subArr.data(), subSize, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else {
            // Receive the subarray from process 0
            MPI_Scatter(NULL, subSize, MPI_INT, subArr.data(), subSize, MPI_INT, 0, MPI_COMM_WORLD);
        }

        double start_time = MPI_Wtime();
        mergeSort(subArr, 0, subSize - 1);
        double end_time = MPI_Wtime();

        std::vector<int> sortedArr(size);
        MPI_Gather(subArr.data(), subSize, MPI_INT, sortedArr.data(), subSize, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // std::cout << "Sorted Array (" << numProcesses << " Processes): ";
            // for (int i = 0; i < size; ++i)
                // std::cout << sortedArr[i] << " ";
            // std::cout << std::endl;

            std::cout << "Execution Time (" << numProcesses << " Processes): " << end_time - start_time << " seconds" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
