#include <iostream>
#include <ctime>
#include <omp.h>

int main()
{
    uint32_t size = 1e6;
    int32_t* array = new int32_t[size];

    for (uint32_t i = 0; i < size; ++i)
        array[i] = 1;

    uint8_t exec = 10;

    std::cout << "Method\tAvrTime(seconds)\n";

    double seq_time = 0.0;
    for (uint8_t j = 0; j < exec; ++j)
    {
        double start = omp_get_wtime();
        int32_t sum_seq = 0;
        for (uint32_t i = 0; i < size; ++i)
            sum_seq += array[i];
        double stop = omp_get_wtime();
        seq_time += (stop - start) / exec;

        // std::cout << "Seq_" << j << ":\t" << (stop - start) << '\n';
    }
    std::cout << "Seq:\t" << seq_time << '\n';

    double par_time = 0.0;
    for (uint8_t j = 0; j < exec; ++j)
    {
        double start = omp_get_wtime();
        int32_t sum_par = 0;
        #pragma omp parallel for reduction(+:sum_par)
        for (uint32_t i = 0; i < size; ++i)
            sum_par += array[i];
        double stop = omp_get_wtime();
        par_time += (stop - start) / exec;

        // std::cout << "Par_" << j << ":\t" << (stop - start) << '\n';
    }
    std::cout << "Par:\t" << par_time << '\n';

    double par_red_time = 0.0;
    for (uint8_t j = 0; j < exec; ++j)
    {
        double start = omp_get_wtime();
        int32_t sum_par_red = 0;
        #pragma omp parallel for reduction(+:sum_par_red)
        for (uint32_t i = 0; i < size; ++i)
            sum_par_red += array[i];
        double stop = omp_get_wtime();
        par_red_time += (stop - start) / exec;

        // std::cout << "Red_" << j << ":\t" << (stop - start) << '\n';
    }
    std::cout << "Red:\t" << par_red_time << '\n';

    delete[] array;

    return 0;
}
