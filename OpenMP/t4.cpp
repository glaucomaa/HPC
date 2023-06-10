#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <omp.h>

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

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                mergeSort(arr, left, mid);
            }

            #pragma omp section
            {
                mergeSort(arr, mid + 1, right);
            }
        }

        merge(arr, left, mid, right);
    }
}

int main() {
    int size = 10000000; // 6.32752 seconds
    int rangeMin = 0;    
    int rangeMax = 1000000;
    std::vector<int> arr(size);

    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < size; ++i)
        arr[i] = rand() % (rangeMax - rangeMin + 1) + rangeMin;

    // std::cout << "Unsorted Array: ";
    // for (int i = 0; i < size; ++i)
        // std::cout << arr[i] << " ";
    // std::cout << std::endl;

    double start_time = omp_get_wtime();
    mergeSort(arr, 0, size - 1);
    double end_time = omp_get_wtime();

    // std::cout << "Sorted Array: ";
    // for (int i = 0; i < size; ++i)
        // std::cout << arr[i] << " ";
    // std::cout << std::endl;

    std::cout << "Execution Time: " << end_time - start_time << " seconds" << std::endl;

    return 0;
}
