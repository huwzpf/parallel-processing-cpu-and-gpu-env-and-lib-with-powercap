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

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define ENABLE_LOGGING
#define MPI_LOGGING
#include "logger.h"
#include "rnn_defines.h"
#include "mkl.h"
#include "string.h"

// Function for element-wise activation (ReLU as an example)
void apply_activation(double* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = fmax(0.0, data[i]); // ReLU activation
    }
}

// RNN function using BLAS
void run_rnn_cpu(double* input, double* hidden_layer_buffer, double* output, 
                 double* weights_input_hidden, double* weights_hidden_hidden, 
                 double* weights_hidden_output, int batch_size) {

    double* current_input;
    double* current_hidden;
    double* next_hidden;
    double* current_output;
    // Initialize hidden state to zero
    memset(hidden_layer_buffer, 0, 2 * batch_size * RNN_HIDDEN_SIZE * sizeof(double));

    for (int t = 0; t < RNN_TIME_STEPS; ++t) {
        // Get pointers to current input, hidden, and output slices
        current_input = input + t * batch_size * RNN_INPUT_SIZE;
        current_hidden = hidden_layer_buffer + (t % 2) * batch_size * RNN_HIDDEN_SIZE;
        next_hidden = hidden_layer_buffer + ((t + 1) % 2) * batch_size * RNN_HIDDEN_SIZE;
        current_output = output + t * batch_size * RNN_OUTPUT_SIZE;

        // Compute next hidden state: H_t = ReLU(W_ih * X_t + W_hh * H_{t-1})
        // temp_hidden = W_ih * X_t
        
        //log_message(LOG_INFO, "Running first");
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    batch_size, RNN_HIDDEN_SIZE, RNN_INPUT_SIZE, 
                    1.0, 
                    current_input, RNN_INPUT_SIZE, 
                    weights_input_hidden, RNN_HIDDEN_SIZE, 
                    0.0, 
                    next_hidden, RNN_HIDDEN_SIZE);

        // temp_hidden += W_hh * H_{t-1}
        if (t > 0) {
            
            //log_message(LOG_INFO, "Running second");
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        batch_size, RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE, 
                        1.0, 
                        current_hidden, RNN_HIDDEN_SIZE, 
                        weights_hidden_hidden, RNN_HIDDEN_SIZE, 
                        1.0, 
                        next_hidden, RNN_HIDDEN_SIZE);
        }

        // Apply activation to the hidden state
            //log_message(LOG_INFO, "Apply activation");
        apply_activation(next_hidden, batch_size * RNN_HIDDEN_SIZE);

        // Compute output: O_t = W_ho * H_t
            //log_message(LOG_INFO, "Running third");
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    batch_size, RNN_OUTPUT_SIZE, RNN_HIDDEN_SIZE, 
                    1.0, 
                    next_hidden, RNN_HIDDEN_SIZE, 
                    weights_hidden_output, RNN_OUTPUT_SIZE, 
                    0.0, 
                    current_output, RNN_OUTPUT_SIZE);
    }
}

extern void launchcpukernel(void *devPtr, unsigned long batchSize, int num_threads) 
{
  mkl_enable_instructions(MKL_ENABLE_SSE4_2);
  mkl_set_num_threads_local(num_threads);
  log_message(LOG_DEBUG, "Launichng CPU Kernel with %llu elements and %i threads.", batchSize, mkl_get_max_threads());
  double *devPtra = (double *)(((void **)devPtr)[0]);
  double *devPtrc = (double *)(((void **)devPtr)[1]);
  double *devPtrk = (double *)(((void **)devPtr)[2]);
  double *devPtrb = (double *)(((void **)devPtr)[3]);
  //log_message(LOG_INFO, "Got devPtr=%lld, a = %lld, c = %lld, k = %lld, b = %lld", devPtr, devPtra, devPtrc, devPtrk, devPtrb);
  run_rnn_cpu(devPtra, devPtrb, devPtrc, devPtrk, devPtrk, devPtrk, batchSize);
}