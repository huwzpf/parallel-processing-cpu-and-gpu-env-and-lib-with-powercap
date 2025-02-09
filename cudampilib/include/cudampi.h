/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#define __cudampi__CUDAMPIMALLOCREQ 1
#define __cudampi__CUDAMPIMALLOCRESP 2
#define __cudampi__CUDAMPICOPYTOGPU 3
#define __cudampi__CUDAMPICOPYFROMGPU 4
#define __cudampi__CUDAMPIFREEREQ 5
#define __cudampi__CUDAMPIFREERESP 6
#define __cudampi__CUDAMPISETDEVICEREQ 7
#define __cudampi__CUDAMPISETDEVICERESP 8
#define __cudampi__CUDAMPIHOSTTODEVICEREQ 9
#define __cudampi__CUDAMPIHOSTTODEVICERESP 10
#define __cudampi__CUDAMPIDEVICETOHOSTREQ 11
#define __cudampi__CUDAMPIDEVICETOHOSTRESP 12
#define __cudampi__CUDAMPILAUNCHCUDAKERNELREQ 13
#define __cudampi__CUDAMPILAUNCHCUDAKERNELRESP 14
#define __cudampi__CUDAMPIDEVICESYNCHRONIZEREQ 15
#define __cudampi__CUDAMPIDEVICESYNCHRONIZERESP 16
#define __cudampi__CUDAMPISTREAMCREATEREQ 17
#define __cudampi__CUDAMPISTREAMCREATERESP 18
#define __cudampi__CUDAMPISTREAMDESTROYREQ 19
#define __cudampi__CUDAMPISTREAMDESTROYRESP 20
#define __cudampi__CUDAMPILAUNCHKERNELINSTREAMREQ 21
#define __cudampi__CUDAMPIHOSTTODEVICEASYNCREQ 23
#define __cudampi__CUDAMPIHOSTTODEVICEASYNCRESP 24
#define __cudampi__CUDAMPIDEVICETOHOSTASYNCREQ 25
#define __cudampi__CUDAMPIDEVICETOHOSTASYNCRESP 26
#define __cudampi__CUDAMPICPUDEVICESYNCHRONIZEREQ 27
#define __cudampi__CUDAMPICPUDEVICESYNCHRONIZERESP 28
#define __cudampi__CPULAUNCHKERNELREQ 29
#define __cudampi__CPUMALLOCREQ 31
#define __cudampi__CPUMALLOCRESP 32
#define __cudampi__CPUFREEREQ 33
#define __cudampi__CPUFREERESP 33
#define __cudampi__CPUHOSTTODEVICEREQ 34
#define __cudampi__CPUHOSTTODEVICERESP 35
#define __cudampi__CPUDEVICETOHOSTREQ 36
#define __cudampi__CPUDEVICETOHOSTRESP 37
#define __cudampi__CPUHOSTTODEVICEREQASYNC 38
#define __cudampi__CPUDEVICETOHOSTREQASYNC 39
#define __cudampi__CPUSTREAMCREATEREQ 40
#define __cudampi__CPUSTREAMCREATERESP 41
#define __cudampi__CPUSTREAMDESTROYREQ 42
#define __cudampi__CPUSTREAMDESTROYRESP 43
#define __cudampi__CONFIGUREPOWERCAP 44
#define __cudampi__CUDAMPIFINALIZE 100
#define MIN_ASYNC_MSG_TAG 101 
#define D_MSG_TAG 2
#define MAX_GPU_PER_NODE 16
#define MAX_ASYNC_MSG_TAG 32000 // According to MPI spec tag upper bound is guaranteed to be no less than 32767 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf page 36, chapter 3.2.3)
#define CPU_STREAMS_SUPPORTED 2
// + 1 stream for asynchronously processing GPU responses to master
#define ALL_CPU_STREAMS CPU_STREAMS_SUPPORTED + MAX_GPU_PER_NODE
// Threads for GPU control + CPU control + asynchronous CPU processing
#define MAX_THREADS MAX_GPU_PER_NODE + 1 + ALL_CPU_STREAMS
