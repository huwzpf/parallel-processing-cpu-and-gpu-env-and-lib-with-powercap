// collatz_legion_dynamic.cu
// ------------------------------------------------------------
// Dynamic CPU+GPU Collatz checker with Legion / Realm.
// ------------------------------------------------------------

/*
export GASNET_PREFIX=$HOME/gasnet-mpi
export GASNET_ROOT=$GASNET_PREFIX
export LEGION_ROOT=~/legion
export LG_RT_DIR=$LEGION_ROOT/runtime
export PATH=$GASNET_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$GASNET_ROOT/lib:$LD_LIBRARY_PATH
export MPIRUN_CMD='mpirun --mca orte_keep_fqdn_hostnames t --mca btl_tcp_if_exclude docker0,docker_gwbridge,lo,vboxnet0 -np %N -hostfile  /home/macierz/s184297/parallel-processing-cpu-and-gpu-env-and-lib-with-powercap/cudampilib/apps/legion/hostfile %C'
export GASNET_SUPERNODE_MAXSIZE=1

gasnetrun_mpi -E LD_LIBRARY_PATH,GASNET_SUPERNODE_MAXSIZE -n 10 collatz_legion -ll:cpu 24 -ll:gpu 1 -ll:csize 30000 -ll:fsize 8000
*/
#include "legion.h"
#include "default_mapper.h"

#include <cassert>
#include <cstdio>
#include <cinttypes>
#include <chrono>
#include <cmath>

using namespace Legion;
using namespace Legion::Mapping;

template<typename FT, int N, typename T = coord_t>
using AccessorRO = FieldAccessor<READ_ONLY,  FT, N, T,
                                 Realm::AffineAccessor<FT, N, T>>;

template<typename FT, int N, typename T = coord_t>
using AccessorWO = FieldAccessor<WRITE_ONLY, FT, N, T,
                                 Realm::AffineAccessor<FT, N, T>>;

static const int64_t DEFAULT_NUM_ELEMENTS = 960000000;
static const int64_t DEFAULT_BATCH_SIZE   =   480000;   // inner task size

enum : FieldID { FID_IN = 0, FID_OUT = 1 };

enum TaskIDs {
  TID_TOP = 0,
  TID_INIT,
  TID_COLLATZ,
};

static inline int isprime(int64_t a) 
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

static inline int64_t collatz_steps(int64_t start) {
  int64_t counter = 0;

  if (isprime(start)) 
  {
    for (; (start > 1); counter++) 
    {
      start = (start % 2) ? (3 * start + 1) : (start / 2);
    }
  }

  return counter;
}

void collatz_cpu_task(const Task*, const std::vector<PhysicalRegion>& regs,
                      Context ctx, Runtime* rt)
{
  AccessorRO<int64_t,1> in (regs[0], FID_IN);
  AccessorWO<int64_t,    1> out(regs[1], FID_OUT);

  Rect<1> r = rt->get_index_space_domain(
                ctx, regs[0].get_logical_region().get_index_space());

  for (PointInRectIterator<1> it(r); it(); ++it)
    out[*it] = collatz_steps(in[*it]);
}

__device__ __forceinline__ int gpu_isprime(int64_t a)
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

__device__ __forceinline__ int64_t gpu_collatz(int64_t start) {
  int64_t counter = 0;

  if (gpu_isprime(start)) 
  {
    for (; (start > 1); counter++) 
    {
      start = (start % 2) ? (3 * start + 1) : (start / 2);
    }
  }

  return counter;
}

__global__ void collatz_kernel(const int64_t* in, int64_t* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = gpu_collatz(in[i]);
}

void collatz_gpu_task(const Task*, const std::vector<PhysicalRegion>& regs,
                      Context ctx, Runtime* rt)
{
  AccessorRO<int64_t,1> in (regs[0], FID_IN);
  AccessorWO<int64_t,    1> out(regs[1], FID_OUT);

  Rect<1> r = rt->get_index_space_domain(
                ctx, regs[0].get_logical_region().get_index_space());

  size_t strides[1];
  const int64_t* in_ptr  = in.ptr(r, strides);
  int64_t*           out_ptr = out.ptr(r, strides);

  const size_t N = r.volume();
  const int BLOCK = 64;
  const int GRID  = static_cast<int>((N + BLOCK - 1) / BLOCK);

  collatz_kernel<<<GRID, BLOCK>>>(in_ptr, out_ptr, N);
}

void init_input_task(const Task* task, const std::vector<PhysicalRegion>& regs,
                     Context ctx, Runtime* rt)
{
  const int64_t base = *reinterpret_cast<const int64_t*>(task->args);
  AccessorWO<int64_t,1> in(regs[0], FID_IN);

  Rect<1> r = rt->get_index_space_domain(
                ctx, regs[0].get_logical_region().get_index_space());

  for (PointInRectIterator<1> it(r); it(); ++it)
    in[*it] = base + (*it)[0];
}

class DynamicCollatzMapper : public DefaultMapper
{
public:
  DynamicCollatzMapper(Runtime* rt, Machine m, Processor local)
    : DefaultMapper(rt->get_mapper_runtime(), m, local) {}

  void select_task_options(const MapperContext ctx,
                           const Task& task, TaskOptions& o) override
  {
    DefaultMapper::select_task_options(ctx, task, o);
    if (task.task_id == TID_COLLATZ) o.stealable = true;
  }
};

void top_level_task(const Task*, const std::vector<PhysicalRegion>&,
                    Context ctx, Runtime* rt)
{
  int64_t N = DEFAULT_NUM_ELEMENTS, B = DEFAULT_BATCH_SIZE;

  // Build the main LogicalRegion
  Rect<1> full(Point<1>(0), Point<1>(N - 1));
  IndexSpace is = rt->create_index_space(ctx, full);

  FieldSpace fs = rt->create_field_space(ctx);
  {
    FieldAllocator fa = rt->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(int64_t), FID_IN);
    fa.allocate_field(sizeof(int64_t),     FID_OUT);
  }
  LogicalRegion lr = rt->create_logical_region(ctx, is, fs);

  // Initialize input array
  {
    const int64_t start_val = 80000000;
    TaskLauncher init(TID_INIT, TaskArgument(&start_val, sizeof(start_val)));
    init.add_region_requirement(
        RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    init.add_field(0, FID_IN);
    rt->execute_task(ctx, init).wait();
  }

  // Partition into stealable batches
  const int64_t NB = (N + B - 1) / B;           // #batch colors
  Rect<1> colors(Point<1>(0), Point<1>(NB - 1));
  IndexSpace cs = rt->create_index_space(ctx, colors);

  Transform<1,1> xform;  xform[0][0] = B;       // stride between batches
  Rect<1> extent(Point<1>(0), Point<1>(B - 1));

  IndexPartition ip = rt->create_partition_by_restriction(
                          ctx, is, cs, xform, extent);
  LogicalPartition lp = rt->get_logical_partition(ctx, lr, ip);

  // Launch one task per batch
  ArgumentMap amap;
  IndexLauncher il(TID_COLLATZ, cs, TaskArgument(), amap);

  il.add_region_requirement(
      RegionRequirement(lp, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, lr));
  il.add_field(0, FID_IN);

  il.add_region_requirement(
      RegionRequirement(lp, 0,
                        WRITE_DISCARD, EXCLUSIVE, lr));
  il.add_field(1, FID_OUT);

  auto t0 = std::chrono::high_resolution_clock::now();
  FutureMap fm = rt->execute_index_space(ctx, il);
  fm.wait_all_results();
  auto t1 = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration_cast
                <std::chrono::duration<double>>(t1 - t0).count();
  printf("Compute time = %.3f seconds\n", secs);
}

int main(int argc, char** argv)
{
  Runtime::set_top_level_task_id(TID_TOP);

  // top-level 
  {
    TaskVariantRegistrar r(TID_TOP, "top_level");
    r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(r, "top_level_cpu");
  }
  // init
  {
    TaskVariantRegistrar r(TID_INIT, "init_input");
    r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    r.set_leaf();
    Runtime::preregister_task_variant<init_input_task>(r, "init_input_cpu");
  }
  // collatz CPU
  {
    TaskVariantRegistrar r(TID_COLLATZ, "collatz_cpu");
    r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    r.set_leaf();
    Runtime::preregister_task_variant<collatz_cpu_task>(r);
  }
  // collatz GPU 
  {
    TaskVariantRegistrar r(TID_COLLATZ, "collatz_gpu");
    r.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    r.set_leaf();
    Runtime::preregister_task_variant<collatz_gpu_task>(r);
  }

  // custom mapper
  Runtime::add_registration_callback(
    [](Machine m, Runtime* rt, const std::set<Processor>& locals)
    {
      for (Processor p : locals)
        rt->replace_default_mapper(new DynamicCollatzMapper(rt, m, p), p);
    });

  printf("\nLaunching the application\n");
  return Runtime::start(argc, argv);
}