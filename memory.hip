#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <ranges>
#include <stacktrace>

#include "gpu.hpp"
#include "benchmark.hpp"

template<typename T, int block_dim, int items_per_thread>
__global__ __launch_bounds__(block_dim)
void load_kernel(T* __restrict__ buffer) {
    constexpr const auto wim = warpSize;

    const auto bid = blockIdx.x;
    const auto tid = threadIdx.x;
    const auto wid = tid / wim;
    const auto lid = __lane_id();

    auto offset = bid * block_dim * items_per_thread + wid * wim * items_per_thread + lid * 4;
    const auto* ptr = buffer + offset;

    if constexpr (items_per_thread == 16) {
        __uint128_t a, b, c, d;
        asm volatile(
            "global_load_dwordx4 %0, %4 off glc slc\n\t"
            "global_load_dwordx4 %1, %4 off offset:1024 glc slc\n\t"
            "global_load_dwordx4 %2, %4 off offset:2048 glc slc\n\t"
            "global_load_dwordx4 %3, %4 off offset:3072 glc slc\n\t"
            "s_waitcnt vmcnt(0)"
            : "=&v"(a), "=&v"(b), "=&v"(c), "=&v"(d)
            : "v"(ptr)
        );
    } else if constexpr (items_per_thread == 32) {
        __uint128_t a, b, c, d;
        __uint128_t e, f, g, h;
        asm volatile(
            "global_load_dwordx4 %0, %8 off offset:-4096 glc slc\n\t"
            "global_load_dwordx4 %1, %8 off offset:-3072 glc slc\n\t"
            "global_load_dwordx4 %2, %8 off offset:-2048 glc slc\n\t"
            "global_load_dwordx4 %3, %8 off offset:-1024 glc slc\n\t"
            "global_load_dwordx4 %4, %8 off offset:0000 glc slc\n\t"
            "global_load_dwordx4 %5, %8 off offset:1024 glc slc\n\t"
            "global_load_dwordx4 %6, %8 off offset:2048 glc slc\n\t"
            "global_load_dwordx4 %7, %8 off offset:3072 glc slc\n\t"
            "s_waitcnt vmcnt(0)"
            : "=&v"(a), "=&v"(b), "=&v"(c), "=&v"(d), "=&v"(e), "=&v"(f), "=&v"(g), "=&v"(h)
            : "v"(ptr + 1024)
        );
    } else {
        static_assert(false, "unreachable");
    }
}

template<typename T, int block_size, int items_per_thread>
void load(benchmark::executor& exec) {
    const size_t grid_size = exec.dev.properties.totalGlobalMem * 90 / 100 / (sizeof(T) * items_per_thread * block_size);
    const size_t buffer_items = grid_size * block_size * items_per_thread;

    const auto buffer_size = benchmark::size(buffer_items);
    const auto buffer_size_bytes = buffer_size.to_bytes<T>();
    const auto reads = benchmark::size(grid_size * block_size * items_per_thread);
    const auto read_bytes = reads.to_bytes<T>();

    std::cout << "total gmem:       " << benchmark::size(exec.dev.properties.totalGlobalMem).giga() << " GB\n";
    std::cout << "l2 cache size:    " << exec.dev.properties.l2CacheSize << '\n';
    std::cout << "compute units:    " << exec.dev.properties.multiProcessorCount << '\n';
    std::cout << "grid size:        " << grid_size << '\n';
    std::cout << "block size:       " << block_size << '\n';
    std::cout << "items per thread: " << items_per_thread << '\n';
    std::cout << "buffer size:      " << buffer_size.giga() << " GI\n";
    std::cout << "buffer size:      " << buffer_size_bytes.giga() << " GB\n";
    std::cout << "total reads:      " << reads.giga() << " GI\n";
    std::cout << "total reads:      " << read_bytes.giga() << " GB\n";

    const gpu::launch_config cfg = {
        .grid_size = grid_size,
        .block_size = block_size,
    };

    const auto buffer = exec.dev.alloc<T>(buffer_items);
    const auto stats = exec.bench([&](const auto& stream) {
        stream.launch(cfg, load_kernel<T, block_size, items_per_thread>, buffer.raw);
    });

    std::cout << "time per launch: " << std::chrono::duration_cast<std::chrono::microseconds>(stats.average)
        << " +- " << std::chrono::duration_cast<std::chrono::microseconds>(stats.stddev) << "\n";
    std::cout << "throughput:      " << benchmark::throughput(reads, stats.average).giga() << " Gitems/s\n";
    std::cout << "throughput:      " << benchmark::throughput(read_bytes, stats.average).giga() << " GB/s\n";
    std::cout << '\n';
}

int main() {
    try {
        const auto dev = gpu::get_default_device();
        auto exec = benchmark::executor(dev);

        load<int, 64, 16>(exec);
        load<int, 64, 32>(exec);
        load<int, 128, 16>(exec);
        load<int, 128, 32>(exec);
        load<int, 256, 16>(exec);
        load<int, 256, 32>(exec);
        load<int, 512, 16>(exec);
        load<int, 512, 32>(exec);
        load<int, 1024, 16>(exec);
        load<int, 1024, 32>(exec);
    } catch (const gpu::error& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::cerr << e.trace << "\n";
        std::exit(1);
    } catch (const std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::exit(1);
    }
}
