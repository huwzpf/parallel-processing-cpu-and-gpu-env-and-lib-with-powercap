//  Copyright (c) 2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstdint>



__device__ __forceinline__ int gpu_isprime(int64_t a)
{
  long i;
  for (i = 2; i < sqrt((double)a) + 1; i++) 
  {
    if ((a % i) == 0) 
    {
      return 0;
    }
  }
  return 1;
}

__global__ void kernel(const std::int64_t* in,
                               std::int64_t*       out,
                               std::size_t         n)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    std::int64_t start = in[idx];
    std::int64_t counter = 0;

    if (gpu_isprime(start)) 
    {
        for (; (start > 1); counter++) 
        {
        start = (start % 2) ? (3 * start + 1) : (start / 2);
        }
    }

    out[idx] = counter;
}

