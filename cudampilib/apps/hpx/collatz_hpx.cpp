//
// HPX build configuration:
//
// cmake .. -DCMAKE_BUILD_TYPE=Release -DHPX_WITH_CUDA=ON -DHPX_WITH_NETWORKING=ON -DHPX_WITH_MULTIPLE_GPUS=ON  -DHPX_WITH_FETCH_ASIO=ON -DHPX_WITH_MALLOC=system -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=~/hpx/install -DHPX_WITH_TOOLS=ON -DHPX_WITH_ASYNC_CUDA=ON -DHPX_WITH_MPI=ON -DHPX_WITH_PARCELPORT_MPI=ON
//
// This app build:
//
// mkdir build && cd build
// cmake .. -DHPX_DIR=$HOME/hpx/install/lib/cmake/HPX
// cmake --build . -j$(nproc)
//
// Run the app with:
//
// mpirun --mca orte_keep_fqdn_hostnames t --mca btl_tcp_if_exclude docker0,docker_gwbridge,lo,vboxnet0 -np 10 -hostfile  /home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/apps/legion/hostfile ./collatz --hpx:threads=all --hpx:run-hpx-main

#include <hpx/hpx_main.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/collectives.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/async_cuda.hpp>

#include <atomic>
#include <vector>
#include <chrono>
#include <numeric>

struct batch_counter : hpx::components::component_base<batch_counter>
{
    batch_counter() = default;
    explicit batch_counter(std::size_t limit) : limit_(limit) {}

    std::size_t fetch_next()
    {
        auto id = next_.fetch_add(1, std::memory_order_relaxed);
        return id < limit_ ? id : limit_;
    }

    HPX_DEFINE_COMPONENT_ACTION(batch_counter, fetch_next);

private:
    std::atomic<std::size_t> next_{0};
    std::size_t              limit_{0};
};
HPX_REGISTER_COMPONENT(hpx::components::component<batch_counter>, batch_counter);
HPX_REGISTER_ACTION(batch_counter::fetch_next_action);


struct collatz_data : hpx::components::component_base<collatz_data>
{
    collatz_data() = default;

    explicit collatz_data(std::size_t N) : input_(N), output_(N)
    {
        std::iota(input_.begin(), input_.end(), 80'000'000);
    }

    // ship one input slice to requester
    std::vector<std::int64_t> get_input(std::size_t begin, std::size_t size) const
    {
        return {input_.begin() + begin, input_.begin() + begin + size};
    }

    // receive back result slice
    void set_output(std::size_t begin, std::vector<std::int64_t> const& slice)
    {
        std::copy(slice.begin(), slice.end(), output_.begin() + begin);
    }

    HPX_DEFINE_COMPONENT_ACTION(collatz_data, get_input);
    HPX_DEFINE_COMPONENT_ACTION(collatz_data, set_output);

private:
    std::vector<std::int64_t> input_, output_;
};
HPX_REGISTER_COMPONENT(hpx::components::component<collatz_data>, collatz_data);
HPX_REGISTER_ACTION(collatz_data::get_input_action);
HPX_REGISTER_ACTION(collatz_data::set_output_action);


static inline int isprime(long a)
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

inline std::int64_t cpu_kernel (std::int64_t start) 
{
    std::int64_t counter = 0;

    if (isprime(start)) 
    {
        for (; (start > 1); counter++) 
        {
        start = (start % 2) ? (3 * start + 1) : (start / 2);
        }
    }

    return counter;
}

// GPU kernel
extern __global__ void kernel(const std::int64_t* in,  std::int64_t* out, std::size_t n);

int main()
{
    constexpr std::size_t N     = 960000000;
    constexpr std::size_t BATCH = 480000;
    const     std::size_t num_batches = (N + BATCH - 1) / BATCH;

    const std::uint32_t this_loc    = hpx::get_locality_id();
    const std::uint32_t num_locales = hpx::get_num_localities(hpx::launch::sync);

    auto comm = hpx::collectives::create_communicator( "collatz/comm", hpx::collectives::num_sites_arg(num_locales), hpx::collectives::this_site_arg(this_loc));

    hpx::id_type counter_id;
    hpx::id_type data_gid;

    if (this_loc == 0)
    {
        counter_id = hpx::new_<batch_counter>(hpx::find_here(), num_batches).get();
        data_gid   = hpx::new_<collatz_data >(hpx::find_here(), N).get();
        hpx::collectives::broadcast_to(comm, counter_id).get();
        hpx::collectives::broadcast_to(comm, data_gid  ).get();
    }
    else
    {
        counter_id = hpx::collectives::broadcast_from<hpx::id_type>(comm).get();
        data_gid   = hpx::collectives::broadcast_from<hpx::id_type>(comm).get();
    }

    batch_counter::fetch_next_action      next_batch;
    collatz_data::get_input_action        fetch_in;
    collatz_data::set_output_action       return_out;

    auto get_batch = [&]{ return hpx::async(next_batch, counter_id).get(); };

    auto cpu_worker = [&]
    {
        std::vector<std::int64_t> in(BATCH), out(BATCH);

        while (true)
        {
            std::size_t id = get_batch();
            if (id >= num_batches) break;

            std::size_t begin = id * BATCH;
            std::size_t size  = std::min(BATCH, N - begin);

            in  = hpx::async(fetch_in , data_gid, begin, size).get();
            out.resize(size);

            hpx::experimental::for_loop(hpx::execution::par, std::size_t{0}, size,
                          [&](std::size_t i){ out[i] = cpu_kernel(in[i]); });

            hpx::async(return_out, data_gid, begin, std::move(out)).get();
        }
    };

    auto gpu_worker = [&]
    {
        hpx::cuda::experimental::enable_user_polling p("default");
        hpx::cuda::experimental::cuda_executor ex(0, hpx::cuda::experimental::event_mode{});

        std::int64_t* d_in = nullptr;
        std::int64_t* d_out= nullptr;
        cudaMalloc(&d_in , BATCH * sizeof(std::int64_t));
        cudaMalloc(&d_out, BATCH * sizeof(std::int64_t));

        std::vector<std::int64_t> host_in(BATCH), host_out(BATCH);
        hpx::cuda::experimental::check_cuda_error(cudaHostRegister(host_in.data(), BATCH * sizeof(std::int64_t), cudaHostRegisterDefault));
        hpx::cuda::experimental::check_cuda_error(cudaHostRegister(host_out.data(), BATCH * sizeof(std::int64_t), cudaHostRegisterDefault));

        while (true)
        {
            std::size_t id = get_batch();
            if (id >= num_batches) break;

            std::size_t begin = id * BATCH;
            std::size_t size  = std::min(BATCH, N - begin);

            host_in = hpx::async(fetch_in, data_gid, begin, size).get();
            hpx::post(ex, cudaMemcpyAsync,d_in, host_in.data(),
                            size*sizeof(std::int64_t), cudaMemcpyHostToDevice);

            int blocks = static_cast<int>((size + 63)/64);
            void* args[] = { &d_in, &d_out, &size };
            hpx::post(ex, cudaLaunchKernel<void>,
                      reinterpret_cast<void*>(&kernel),
                      dim3(blocks), dim3(64), args, 0);

            hpx::async(ex, cudaMemcpyAsync, host_out.data(), d_out,
                       size*sizeof(std::int64_t), cudaMemcpyDeviceToHost).get();

            hpx::async(return_out, data_gid, begin, host_out).get();
        }
        cudaFree(d_in); cudaFree(d_out);
    };


    auto t0 = std::chrono::high_resolution_clock::now();
    hpx::future<void> cpu_f = hpx::async(cpu_worker);
    hpx::future<void> gpu_f = hpx::async(gpu_worker);
    hpx::wait_all(cpu_f, gpu_f);


    hpx::distributed::barrier("/collatz/barrier").wait();

    if (this_loc == 0)
    {
        double secs = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        hpx::cout << "Processed " << N << " numbers in "
                  << secs << " s ("
                  << num_locales << " localities)\n";
    }
}
