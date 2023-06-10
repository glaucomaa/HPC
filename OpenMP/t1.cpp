#include <iostream>
#include <ctime>
#include <omp.h>

int main()
{
    const uint32_t numIterations = 1e6;
    const uint8_t numExecutions = 10;

    double seqSum, parSum;
    double seqTime, parTime;

    double seqStart, seqStop;
    double parStart, parStop;

    seqStart = omp_get_wtime();
    for (uint8_t j = 0; j < numExecutions; ++j)
    {
        seqSum = 0;
        for (uint32_t i = 0; i < numIterations; ++i)
        {
            double x = (i + 0.5) / numIterations;
            seqSum += 4 / (1 + x * x);
        }
        seqSum /= numIterations;
    }
    seqStop = omp_get_wtime();
    seqTime = (seqStop - seqStart) / numExecutions;

    parStart = omp_get_wtime();
    for (uint8_t j = 0; j < numExecutions; ++j)
    {
        parSum = 0;
#pragma omp parallel
        {
            double localSum = 0;
#pragma omp for
            for (uint32_t i = 0; i < numIterations; ++i)
            {
                double x = (i + 0.5) / numIterations;
                localSum += 4 / (1 + x * x);
            }
#pragma omp atomic
            parSum += localSum;
        }
        parSum /= numIterations;
    }
    parStop = omp_get_wtime();
    parTime = (parStop - parStart) / numExecutions;

    std::cout.precision(10);
    std::cout << "Method\tAverage Time (sec)\tResult\n";
    std::cout << "Sequential\t" << seqTime << '\t';
    std::cout << seqSum << '\n';
    std::cout << "Parallel\t" << parTime << '\t';
    std::cout << parSum << '\n';

    return 0;
}
