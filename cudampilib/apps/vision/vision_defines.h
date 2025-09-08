/*
Simple vision app parameters: CPU does preprocessing, GPU runs a conv layer.
*/
#ifndef VISION_DEFINES_H
#define VISION_DEFINES_H

// Problem sizing
#define VISION_NUM_SAMPLES        20000   // total number of images
#define VISION_IMG_C              3       // RGB
#define VISION_IMG_H              64
#define VISION_IMG_W              64
#define VISION_K                  3       // 3x3 conv kernel

// Single-output-channel conv to keep it lightweight
#define VISION_OUT_C              1

// Derived dims
#define VISION_OUT_H              (VISION_IMG_H - VISION_K + 1)
#define VISION_OUT_W              (VISION_IMG_W - VISION_K + 1)

// Per-sample sizes
#define VISION_INPUT_SIZE         (VISION_IMG_C * VISION_IMG_H * VISION_IMG_W)
#define VISION_OUTPUT_SIZE        (VISION_OUT_C * VISION_OUT_H * VISION_OUT_W)

// Weights size (OUT_C x C x K x K)
#define VISION_WEIGHTS_SIZE       (VISION_OUT_C * VISION_IMG_C * VISION_K * VISION_K)

// Execution config
#define VISION_THREADS_IN_BLOCK   256

#endif

