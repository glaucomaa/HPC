#include <iostream>
#include <iomanip>
#include <ctime>
#include <omp.h>
#include <vector>

double calculatePi(uint32_t N, uint8_t exec)
{
    double par_sum = 0;

    for (uint8_t j = 0; j < exec; ++j)
    {
        double x;
        double sum = 0;

        double par_start = omp_get_wtime();

#pragma omp parallel private(x) reduction(+: sum)
        {
            uint32_t thread_count = omp_get_num_threads();
            uint32_t thread_id = omp_get_thread_num();

            for (uint32_t i = thread_id; i < N; i += thread_count)
            {
                x = (i + 0.5) / N;
                sum += 4 / (1 + x * x);
            }
        }

        par_sum += sum;
        double par_stop = omp_get_wtime();
    }

    par_sum /= (N * exec);

    return par_sum;
}

int main()
{
    double start_time, end_time, execution_time;  

    uint32_t N_small = 1e2, N_big = 1e6;
    std::vector<uint8_t> exec_values = {1, 2, 4, 8, 10, 12};

    std::cout << std::setprecision(8) << std::fixed;
    std::cout << "Threads\tN\tTime\t\tResult\n";
    
    for (uint8_t exec : exec_values)
    {
        start_time = omp_get_wtime();
        double pi_result = calculatePi(N_small, exec);
        end_time = omp_get_wtime();
        execution_time = (end_time - start_time) / exec;

        std::cout << static_cast<int>(exec) << "\t1e2\t" << execution_time << '\t' << pi_result << '\n';
        
        start_time = omp_get_wtime();
        pi_result = calculatePi(N_big, exec);
        end_time = omp_get_wtime();
        execution_time = (end_time - start_time) / exec;

        std::cout << static_cast<int>(exec) << "\t1e6\t" << execution_time << '\t' << pi_result << '\n';

        std::cout << std::endl;
    }

    return 0;
}
