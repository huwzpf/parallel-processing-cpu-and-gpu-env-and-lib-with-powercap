/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "cudampilib.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define ENABLE_LOGGING
#include "logger.h"

int main(int argc, char **argv) {

  __cudampi__initializeMPI(argc, argv);

  int cudadevicescount = 1;

  __cudampi__getDeviceCount(&cudadevicescount);

  #pragma omp parallel num_threads(cudadevicescount)
  {

    void *devPtr;
    unsigned char tab[4096];
    int i;

    int mythreadid = omp_get_thread_num();

    for (i = 0; i < 2048; i++) {
      tab[i] = i;
    }

    __cudampi__setDevice(mythreadid);

    __cudampi__malloc(&devPtr, 4096);

    __cudampi__memcpy(devPtr, tab, 2048, cudaMemcpyHostToDevice);

    __cudampi__kernel(devPtr, 0);

    __cudampi__memcpy(tab + 2048, devPtr, 2048, cudaMemcpyDeviceToHost);

    __cudampi__deviceSynchronize();

    for (i = 2048; i < 4096; i++) {
      log_message(LOG_DEBUG, "\n ind[%d]=%d", i, (int)(tab[i]));
    }

    __cudampi__free(devPtr);
  }

  __cudampi__terminateMPI();
}
