/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
// idea: in most cases, input data can be uploaded to GPU's memory and
// consequently we only need to copy a pointer in kernel invocation
// in OpenCL we could hide any kernel invocation

#include <stdio.h>
#include <math.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "twinprime_defines.h"

int isprime(long long a) 
{
  long long i;
  for (i = 2; i < sqrt((double)a) + 1; i++) 
  {
    if ((a % i) == 0) 
    {
      return 0;
    }
  }
  return 1;
}

void appkernel(void *devPtr, long long num_elements, int num_threads) 
{
    long long *input = (long long *)(((void **)devPtr)[0]);
    long long *output = (long long *)(((void **)devPtr)[1]);

    #pragma omp parallel for num_threads(num_threads)
    for (long long i = 0; i < num_elements; i++) 
    {
        if (isprime(input[i]) && isprime(input[i + 2]) && (input[i] != input[i + 2])) 
        {
            output[i] = 1;
        } else {
            output[i] = 0;
        }
    }
}

extern void launchcpukernel(void *devPtr, unsigned long batchSize, int num_threads, unsigned long long /* id */) 
{
  long long num_elements = batchSize;
  log_message(LOG_DEBUG, "Launichng CPU Kernel with %i elements and %i threads.", num_elements, num_threads);
  appkernel(devPtr, num_elements, num_threads);
}