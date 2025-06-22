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

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>

#define ENABLE_LOGGING_GPU
#define ENABLE_LOGGING
#include "logger_gpu.h"
#include "logger.h"
#include "rnn_defines.h"

// CUDA kernel for element-wise activation (ReLU as an example)
__global__ void apply_activation(double* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmax(0.0, data[idx]); // ReLU activation
    }
}

// RNN function using cuBLAS
void run_rnn(void* devPtr, unsigned long batch_size) {
    
    // It is not very efficient but not sure how to do it otherwise
    void** new_ptr[4];
    cudaMemcpy(new_ptr, devPtr, 4*sizeof(void*), cudaMemcpyDeviceToHost);
    double *input = (double *)(((void **)new_ptr)[0]);
    double *output = (double *)(((void **)new_ptr)[1]);
    double *devPtrk = (double *)(((void **)new_ptr)[2]);
    double *hidden_layer_buffer = (double *)(((void **)new_ptr)[3]);
    double* weights_input_hidden = devPtrk;
    double* weights_hidden_hidden = devPtrk;
    double* weights_hidden_output = devPtrk;

    cublasHandle_t handle;
    cublasCreate(&handle);

    double *current_input, *current_hidden, *next_hidden, *current_output;
    double alpha = 1.0, beta = 0.0;

    // Initialize hidden state to zero
    cudaMemset(hidden_layer_buffer, 0, batch_size * RNN_HIDDEN_SIZE * sizeof(double));

    for (int t = 0; t < RNN_TIME_STEPS; ++t) {
        // Get pointers to current input, hidden, and output slices
        current_input = input + t * batch_size * RNN_INPUT_SIZE;
        current_hidden = hidden_layer_buffer + (t % 2) * batch_size * RNN_HIDDEN_SIZE;
        next_hidden = hidden_layer_buffer + ((t + 1) % 2) * batch_size * RNN_HIDDEN_SIZE;
        current_output = output + t * batch_size * RNN_OUTPUT_SIZE;

        // Compute next hidden state: H_t = ReLU(W_ih * X_t + W_hh * H_{t-1} + b_h)
        // temp_hidden = W_ih * X_t
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    RNN_HIDDEN_SIZE, batch_size, RNN_INPUT_SIZE, 
                    &alpha, 
                    weights_input_hidden, RNN_HIDDEN_SIZE, 
                    current_input, RNN_INPUT_SIZE, 
                    &beta, 
                    next_hidden, RNN_HIDDEN_SIZE);

        // temp_hidden += W_hh * H_{t-1}
        if (t > 0) {
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        RNN_HIDDEN_SIZE, batch_size, RNN_HIDDEN_SIZE, 
                        &alpha, 
                        weights_hidden_hidden, RNN_HIDDEN_SIZE, 
                        current_hidden, RNN_HIDDEN_SIZE, 
                        &alpha, 
                        next_hidden, RNN_HIDDEN_SIZE);
        }

        // Add bias to hidden state
        dim3 blockSize(RNN_THREADS_IN_BLOCK);
        dim3 gridSize((batch_size * RNN_HIDDEN_SIZE + blockSize.x - 1) / blockSize.x);
        apply_activation<<<gridSize, blockSize>>>(next_hidden, batch_size * RNN_HIDDEN_SIZE);

        // Compute output: O_t = W_ho * H_t + b_o
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    RNN_OUTPUT_SIZE, batch_size, RNN_HIDDEN_SIZE, 
                    &alpha, 
                    weights_hidden_output, RNN_OUTPUT_SIZE, 
                    next_hidden, RNN_HIDDEN_SIZE, 
                    &beta, 
                    current_output, RNN_OUTPUT_SIZE);

    }

    cublasDestroy(handle);
}

extern "C" void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream, unsigned long long id) 
{
  run_rnn(devPtr,  batchSize);
  cudaError_t e = cudaGetLastError();
  if (cudaSuccess != e) {
    log_message(LOG_ERROR, "Error during kernel launch in stream, %s", cudaGetErrorString(e));
  }
}

extern "C" void launchkernel(void *devPtr, unsigned long batchSize, unsigned long long id) { launchkernelinstream(devPtr, batchSize, 0, id); }
