/*
Simple multi-layer CNN forward pass on GPU.
Uses same conv weights for all CNN layers; final FC classifier.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cudnn.h>

#define ENABLE_LOGGING_GPU
#define ENABLE_LOGGING
#include "logger_gpu.h"
#include "logger.h"
#include "cnn_defines.h"

// devPtr is a device pointer to an array of 4 device pointers:
// [0] input  (double*)  size: B * INPUT_BATCH_SIZE
// [1] output (double*)  size: B * OUTPUT_BATCH_SIZE
// [2] weights(double*)  size: WEIGHTS_SIZE
// [3] scratch(double*)  size: B * 2 * FEATURE_SIZE (ping-pong)
static void run_cnn(void* devPtr, unsigned long B, cudaStream_t stream)
{
    void* host_ptrs[5];
    cudaMemcpy(host_ptrs, devPtr, sizeof(host_ptrs), cudaMemcpyDeviceToHost);
    float* input   = (float*)host_ptrs[0];
    float* output  = (float*)host_ptrs[1];
    float* weights = (float*)host_ptrs[2];
    float* scratch = (float*)host_ptrs[3];
    void*  workspace = host_ptrs[4];

    const int Cin = CNN_IN_CHANNELS;
    const int Ch  = CNN_HIDDEN_CHANNELS;
    const int H = CNN_IMG_H;
    const int W = CNN_IMG_W;
    const int K = CNN_KERNEL_SIZE;
    const int L = CNN_LAYERS;
    const int CLS = CNN_CLASSES;

    float* w_conv0 = weights;                                // [Ch, Cin, K, K]
    float* w_convH = weights + CONV0_WEIGHTS_SIZE;           // [Ch, Ch,  K, K]
    float* w_fc    = weights + CONV_WEIGHTS_SIZE;            // [CLS, Ch, 1, 1]

    // Ping-pong feature storage in scratch
    long feat_elems_total = (long)B * FEATURE_SIZE; // B*Ch*H*W
    float* ping = scratch;                          // [B, Ch, H, W]
    float* pong = scratch + feat_elems_total;       // [B, Ch, H, W]

    // cuDNN setup
    cudnnHandle_t handle; cudnnCreate(&handle); cudnnSetStream(handle, stream);

    cudnnTensorDescriptor_t inDesc0, hidDesc;
    cudnnFilterDescriptor_t filt0Desc, filthDesc;
    cudnnConvolutionDescriptor_t conv0Desc, convHDesc;
    cudnnActivationDescriptor_t actDesc;
    cudnnPoolingDescriptor_t poolDesc;

    cudnnCreateTensorDescriptor(&inDesc0);
    cudnnCreateTensorDescriptor(&hidDesc);
    cudnnCreateFilterDescriptor(&filt0Desc);
    cudnnCreateFilterDescriptor(&filthDesc);
    cudnnCreateConvolutionDescriptor(&conv0Desc);
    cudnnCreateConvolutionDescriptor(&convHDesc);
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnCreatePoolingDescriptor(&poolDesc);

    // Descriptors (Cin -> Ch for first layer, then Ch -> Ch)
    cudnnSetTensor4dDescriptor(inDesc0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)B, Cin, H, W);
    cudnnSetTensor4dDescriptor(hidDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)B, Ch,  H, W);
    cudnnSetFilter4dDescriptor(filt0Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, Ch, Cin, K, K);
    cudnnSetFilter4dDescriptor(filthDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, Ch, Ch,  K, K);
    int pad = K / 2; // SAME padding
    cudnnSetConvolution2dDescriptor(conv0Desc, pad, pad, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolution2dDescriptor(convHDesc, pad, pad, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
#if CNN_USE_TENSOR_OPS
    cudnnSetConvolutionMathType(conv0Desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
    cudnnSetConvolutionMathType(convHDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
#endif
    cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    // Pick fixed algorithms to avoid runtime fluctuations
    cudnnConvolutionFwdAlgo_t algo0 = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    cudnnConvolutionFwdAlgo_t algoH = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t ws0 = 0, wsH = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, inDesc0, filt0Desc, conv0Desc, hidDesc, algo0, &ws0);
    cudnnGetConvolutionForwardWorkspaceSize(handle, hidDesc,  filthDesc, convHDesc, hidDesc, algoH, &wsH);
    size_t ws_need = ws0 > wsH ? ws0 : wsH;
    size_t ws_cap  = (size_t)CNN_WORKSPACE_MB * 1024ULL * 1024ULL;
    if (ws_need > ws_cap) {
        // Fallback to low-workspace algos if requested exceeds capacity
        algo0 = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        algoH = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        cudnnGetConvolutionForwardWorkspaceSize(handle, inDesc0, filt0Desc, conv0Desc, hidDesc, algo0, &ws0);
        cudnnGetConvolutionForwardWorkspaceSize(handle, hidDesc,  filthDesc, convHDesc, hidDesc, algoH, &wsH);
        ws_need = ws0 > wsH ? ws0 : wsH;
    }
    size_t ws_bytes = (ws_need <= ws_cap) ? ws_need : 0; // ensure not to exceed capacity

    const float alpha = 1.0f, beta0 = 0.0f;

    // First layer: input -> ping, then ReLU
    cudnnConvolutionForward(handle, &alpha,
                            inDesc0, input,
                            filt0Desc, w_conv0,
                            conv0Desc, algo0, workspace, ws_bytes,
                            &beta0,
                            hidDesc, ping);
    cudnnActivationForward(handle, actDesc, &alpha, hidDesc, ping, &beta0, hidDesc, ping);

    // Middle layers: alternate ping/pong with the same filter (shared weights)
    for (int li = 1; li < L; ++li) {
        float* src = (li % 2 == 1) ? ping : pong;
        float* dst = (li % 2 == 1) ? pong : ping;
        cudnnConvolutionForward(handle, &alpha,
                                hidDesc, src,
                                filthDesc, w_convH,
                                convHDesc, algoH, workspace, ws_bytes,
                                &beta0,
                                hidDesc, dst);
        cudnnActivationForward(handle, actDesc, &alpha, hidDesc, dst, &beta0, hidDesc, dst);
    }

    // Select last feature buffer
    float* last_feat = (L == 1) ? ping : ((L % 2 == 0) ? ping : pong);

    // Global Average Pooling: window HxW -> output H=W=1
    cudnnTensorDescriptor_t lastDesc, pooledDesc;
    cudnnCreateTensorDescriptor(&lastDesc);
    cudnnCreateTensorDescriptor(&pooledDesc);
    cudnnSetTensor4dDescriptor(lastDesc,   CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)B, Ch, H, W);
    cudnnSetTensor4dDescriptor(pooledDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)B, Ch, 1, 1);
    cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,
                                H, W, 0, 0, 1, 1);
    float* pooled = pong; // reuse beginning of pong for [B,C,1,1] (size B*C)
    cudnnPoolingForward(handle, poolDesc, &alpha, lastDesc, last_feat, &beta0, pooledDesc, pooled);

    // Fully Connected via 1x1 convolution: input [B,C,1,1], filter [CLS, C, 1,1] -> out [B, CLS, 1,1]
    cudnnFilterDescriptor_t fcFilt;
    cudnnConvolutionDescriptor_t fcConv;
    cudnnTensorDescriptor_t fcOutDesc, fcInDesc;
    cudnnCreateFilterDescriptor(&fcFilt);
    cudnnCreateConvolutionDescriptor(&fcConv);
    cudnnCreateTensorDescriptor(&fcOutDesc);
    cudnnCreateTensorDescriptor(&fcInDesc);
    cudnnSetFilter4dDescriptor(fcFilt, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, CLS, Ch, 1, 1);
    cudnnSetConvolution2dDescriptor(fcConv, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
#if CNN_USE_TENSOR_OPS
    cudnnSetConvolutionMathType(fcConv, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
#endif
    cudnnSetTensor4dDescriptor(fcInDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)B, Ch, 1, 1);
    cudnnSetTensor4dDescriptor(fcOutDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)B, CLS, 1, 1);

    // Fixed algo for FC (1x1 conv)
    cudnnConvolutionFwdAlgo_t fcAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t wsF = 0; cudnnGetConvolutionForwardWorkspaceSize(handle, fcInDesc, fcFilt, fcConv, fcOutDesc, fcAlgo, &wsF);
    if (wsF > ws_cap) {
        fcAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        cudnnGetConvolutionForwardWorkspaceSize(handle, fcInDesc, fcFilt, fcConv, fcOutDesc, fcAlgo, &wsF);
    }
    // For FC we may need less than previous conv allocations; choose min available
    size_t ws_fc_bytes = (wsF <= ws_cap) ? wsF : 0;
    cudnnConvolutionForward(handle, &alpha,
                            fcInDesc, pooled,
                            fcFilt, w_fc,
                            fcConv, fcAlgo, workspace, ws_fc_bytes,
                            &beta0,
                            fcOutDesc, output);

    // Cleanup
    // workspace owned by caller; do not free here
    cudnnDestroyTensorDescriptor(inDesc0);
    cudnnDestroyTensorDescriptor(hidDesc);
    cudnnDestroyTensorDescriptor(lastDesc);
    cudnnDestroyTensorDescriptor(pooledDesc);
    cudnnDestroyTensorDescriptor(fcInDesc);
    cudnnDestroyTensorDescriptor(fcOutDesc);
    cudnnDestroyFilterDescriptor(filt0Desc);
    cudnnDestroyFilterDescriptor(filthDesc);
    cudnnDestroyFilterDescriptor(fcFilt);
    cudnnDestroyConvolutionDescriptor(conv0Desc);
    cudnnDestroyConvolutionDescriptor(convHDesc);
    cudnnDestroyConvolutionDescriptor(fcConv);
    cudnnDestroyActivationDescriptor(actDesc);
    cudnnDestroyPoolingDescriptor(poolDesc);
    cudnnDestroy(handle);
}

extern "C" void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream, unsigned long long /*id*/)
{
    run_cnn(devPtr, batchSize, stream);
    cudaError_t e = cudaGetLastError();
    if (cudaSuccess != e) {
      log_message(LOG_ERROR, "CNN: error after kernel launches: %s", cudaGetErrorString(e));
    }
}

extern "C" void launchkernel(void *devPtr, unsigned long batchSize, unsigned long long id)
{
    launchkernelinstream(devPtr, batchSize, 0, id);
}
