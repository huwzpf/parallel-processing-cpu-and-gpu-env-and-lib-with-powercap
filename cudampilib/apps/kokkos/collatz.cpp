#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

using mem_space  = Kokkos::CudaHostPinnedSpace;
using view_t  = Kokkos::View<std::uint64_t*, mem_space>;
using counter_t  = Kokkos::View<std::uint64_t,  mem_space>;

constexpr std::uint64_t VECTORSIZE_TOTAL   = 960'000'000ULL;
constexpr std::uint64_t BATCH_SIZE         = 480'000ULL;
constexpr int           THREADS_PER_BLOCK  = 64;
constexpr int           GPU_TEAMS          = BATCH_SIZE / THREADS_PER_BLOCK; 

template<class ExecSpace>
struct CollatzPrimeFunctor {
  using member_t   = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

  view_t  in, out;
  counter_t  next;
  const std::uint64_t n_total;
  const std::uint64_t chunk;

  CollatzPrimeFunctor(view_t  in_,
                      view_t  out_,
                      counter_t  next_,
                      std::uint64_t n_total_,
                      std::uint64_t chunk_)
  : in(in_), out(out_), next(next_), n_total(n_total_), chunk(chunk_) {}

  KOKKOS_INLINE_FUNCTION
  bool is_prime(std::uint64_t x) const {
    if (x < 2) return false;
    for (std::uint64_t i = 2; i * i <= x; ++i)
      if (x % i == 0) return false;
    return true;
  }

  KOKKOS_FUNCTION
  void operator()(const member_t& team) const {

    /* every team repeats until global work is exhausted ----------------*/
    while (true) {

      /* team leader grabs the next 480 000-element slice */
      std::uint64_t start;
      if (team.team_rank() == 0)
        start = Kokkos::atomic_fetch_add(&next(), chunk);

      team.team_broadcast(start, 0);
      if (start >= n_total) return;                       // all done

      const std::uint64_t end = (start + chunk > n_total) ?
                                  n_total : start + chunk;

      /* each thread in the team handles strided elements of the slice */
      for (std::uint64_t idx = team.team_rank();
           idx < end - start;
           idx += team.team_size())
      {
        const std::uint64_t i   = start + idx;
        std::uint64_t n         = in(i);
        std::uint64_t steps     = 0;

        if (is_prime(n)) {
          while (n > 1) {
            n = (n & 1ULL) ? 3ULL * n + 1ULL : n >> 1ULL;
            ++steps;
          }
        }
        out(i) = steps;
      }
      team.team_barrier();     // keep teams in lock-step before next slice
    }
  }
};

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Barrier(MPI_COMM_WORLD);
    Kokkos::Timer total_timer;

    const std::uint64_t base = (VECTORSIZE_TOTAL / world) * rank;

    const std::uint64_t local_size = VECTORSIZE_TOTAL / world;
    view_t    in ("input" , local_size);
    view_t    out("output", local_size);
    view_t all_out;
    if (rank == 0) {
      all_out = view_t("all_output", VECTORSIZE_TOTAL);
    }
    counter_t next("next");

    Kokkos::parallel_for("InitInput", Kokkos::RangePolicy<Kokkos::OpenMP>(0, local_size), KOKKOS_LAMBDA(const std::uint64_t i) {
      in(i) = (80'000'000ULL + i + base) % VECTORSIZE_TOTAL;
    });
    Kokkos::deep_copy(next, std::uint64_t(0));
    Kokkos::fence();


    CollatzPrimeFunctor<Kokkos::Cuda> gpu_f(in, out, next, local_size, BATCH_SIZE);
    CollatzPrimeFunctor<Kokkos::OpenMP> cpu_f(in, out, next, local_size, BATCH_SIZE);

    Kokkos::parallel_for("GPUKernel", Kokkos::TeamPolicy<Kokkos::Cuda>(GPU_TEAMS, THREADS_PER_BLOCK), gpu_f);

    Kokkos::parallel_for("CPUKernel", Kokkos::TeamPolicy<Kokkos::OpenMP>(24, Kokkos::AUTO()), cpu_f);

    Kokkos::fence();

    MPI_Barrier(MPI_COMM_WORLD);          

    MPI_Gather(
      /* sendbuf */    out.data(), 
      /* sendcount */  static_cast<int>(local_size),
      /* sendtype */   MPI_UINT64_T,
      /* recvbuf */    all_out.data() + rank * local_size,
      /* recvcount */  local_size,
      /* recvtype */   MPI_UINT64_T,
      /* root */       0,
      /* comm */       MPI_COMM_WORLD
    );
      
    if (rank == 0){
        double local_time = total_timer.seconds();
        printf("Total execution time (max over ranks): %.6f s\n", local_time);
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
