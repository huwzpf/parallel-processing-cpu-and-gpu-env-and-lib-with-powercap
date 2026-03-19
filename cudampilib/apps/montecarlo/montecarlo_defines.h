#ifndef MONTECARLO_DEFINES_H
#define MONTECARLO_DEFINES_H

// Threads per CUDA block
#define MONTECARLO_THREADS_IN_BLOCK 128

// Total number of Monte Carlo work items (distributed in batches)
#define MONTECARLO_PROBLEM_SIZE 960000000ULL

// Number of random point samples computed per work item
#define MONTECARLO_ITERS_PER_ITEM 131072

#endif // MONTECARLO_DEFINES_H

