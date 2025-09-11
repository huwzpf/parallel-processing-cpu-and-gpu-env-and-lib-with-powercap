/*
Simple configurable CNN parameters.
Adjust these macros as needed to change the workload.
*/
#ifndef CNN_DEFINES_H
#define CNN_DEFINES_H

// Image / feature map shape

#define CNN_IN_CHANNELS 1

#define CNN_HIDDEN_CHANNELS 64

#define CNN_IMG_H 256
#define CNN_IMG_W 256

// Convolution
#define CNN_KERNEL_SIZE 7          // square kernel KxK
#define CNN_LAYERS 16              // number of stacked conv layers (same weights for inner layers)

// Classifier
#define CNN_CLASSES 4096

// Total number of elements (samples) to process
#define CNN_VECTORSIZE 4000

// Derived sizes
#define INPUT_BATCH_SIZE   (CNN_IN_CHANNELS * CNN_IMG_H * CNN_IMG_W)
#define FEATURE_SIZE       (CNN_HIDDEN_CHANNELS * CNN_IMG_H * CNN_IMG_W)
#define OUTPUT_BATCH_SIZE  (CNN_CLASSES)

// Sizes for weights: first conv (Cin->Ch), inner conv shared (Ch->Ch), and FC (Ch->CLS)
#define CONV0_WEIGHTS_SIZE  (CNN_HIDDEN_CHANNELS * CNN_IN_CHANNELS   * CNN_KERNEL_SIZE * CNN_KERNEL_SIZE)
#define CONVH_WEIGHTS_SIZE  (CNN_HIDDEN_CHANNELS * CNN_HIDDEN_CHANNELS * CNN_KERNEL_SIZE * CNN_KERNEL_SIZE)
#define CONV_WEIGHTS_SIZE   (CONV0_WEIGHTS_SIZE + CONVH_WEIGHTS_SIZE)
#define FC_WEIGHTS_SIZE     (CNN_HIDDEN_CHANNELS * CNN_CLASSES)
#define WEIGHTS_SIZE        (CONV_WEIGHTS_SIZE + FC_WEIGHTS_SIZE)

// Workspace reserved for cuDNN (per stream), in megabytes
#define CNN_WORKSPACE_MB 256

// Allow cuDNN tensor op math (TF32/FP16 on supported GPUs)
#define CNN_USE_TENSOR_OPS 1

// CPU-side preprocessing filter size for median filter (odd integer >=1)
#define CPU_FILTER_SIZE 9


#endif // CNN_DEFINES_H
