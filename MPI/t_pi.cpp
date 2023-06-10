#include <iostream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <mpi.h>
#include <vector>

double calculatePiParallel(uint64_t num_steps, int rank, int size)
{
    double step = 1.0 / static_cast<double>(num_steps);
    double sum = 0.0;
    
    for (uint64_t i = rank; i < num_steps; i += size)
    {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double local_sum = 0.0;
    MPI_Reduce(&sum, &local_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        return local_sum * step;
    
    return 0.0;
}

double calculatePiSequential(uint64_t num_steps)
{
    double step = 1.0 / static_cast<double>(num_steps);
    double sum = 0.0;
    
    for (uint64_t i = 0; i < num_steps; ++i)
    {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * step;
}

int main(int argc, char** argv)
{
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint64_t num_steps = 1000000000;
    std::vector<int> num_procs_values = {1, 2, 4, 8, 16, 32, 64, 128};

    if (rank == 0)
        std::cout << std::setprecision(15) << std::fixed;

    for (int num_procs : num_procs_values)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time_parallel = MPI_Wtime();
        double pi_parallel = calculatePiParallel(num_steps, rank, num_procs);
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time_parallel = MPI_Wtime();
        double execution_time_parallel = end_time_parallel - start_time_parallel;

        if (rank == 0)
            std::cout << "Processes: " << num_procs << "\n";
            std::cout << "Parallel\t\tPi: " << pi_parallel << "\tTime: " << execution_time_parallel << " seconds" << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            double start_time_sequential = MPI_Wtime();
            double pi_sequential = calculatePiSequential(num_steps);
            double end_time_sequential = MPI_Wtime();
            double execution_time_sequential = end_time_sequential - start_time_sequential;

            std::cout << "Sequential\t\tPi: " << pi_sequential << "\tTime: " << execution_time_sequential << " seconds" << "\n" << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
