export OMPI_CXX=/home/macierz/s184297/kokkos/bin/nvcc_wrapper
export NVCC_WRAPPER_DEFAULT_FLAGS="-arch=sm_89"

mpicxx -O3 -std=c++17 -fopenmp \
       -I/home/macierz/s184297/kokkos/include \
       -L/home/macierz/s184297/kokkos/lib -lkokkoscore \
       -DKOKKOS_ENABLE_CUDA_UVM \
       -arch=sm_89 \
       -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda \
       --extended-lambda \
       -o collatz_kokkos collatz.cpp

mpirun --mca orte_keep_fqdn_hostnames t --mca btl_tcp_if_exclude docker0,docker_gwbridge,lo,vboxnet0 --bind-to none --machinefile ./hostfile -np $1 ./collatz_kokkos

