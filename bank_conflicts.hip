#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <ranges>
#include <stacktrace>
#include <bit>

#include "gpu.hpp"
#include "benchmark.hpp"

constexpr uint32_t block_size = 32;
constexpr uint32_t banks = 32; // TODO: Fetch from architecture config?
constexpr uint32_t words_per_thread = banks * 2; // Should be multiple of number of banks.

using u128 = __uint128_t;
using u96 = int3;
using word = uint32_t;

template<typename T>
__global__ __launch_bounds__(block_size)
void kernel(const int padding_words) {
    constexpr int words_per_item = sizeof(T) / sizeof(word);
    const int stride = words_per_thread + padding_words;

    static_assert(words_per_item > 0, ":(");

    extern __shared__ word x[];

    #if defined(GPU_FAMILY_RDNA4) || defined(GPU_FAMILY_RDNA3)
        #define DS_LOAD_B128 "ds_load_b128"
        #define DS_LOAD_B96 "ds_load_b96"
        #define DS_LOAD_B64 "ds_load_b64"
        #define DS_LOAD_B32 "ds_load_b32"
    #else
        #define DS_LOAD_B128 "ds_read_b128"
        #define DS_LOAD_B96 "ds_read_b96"
        #define DS_LOAD_B64 "ds_read_b64"
        #define DS_LOAD_B32 "ds_read_b32"
    #endif

    #pragma clang loop unroll(full)
    for (int i = 0; i < words_per_thread; i += words_per_item) {
        const auto* addr = &x[threadIdx.x * stride + i];
        const auto* ptr = (const T __attribute__((address_space(3)))*) addr;

        if constexpr (sizeof(T) == 16) {
            asm volatile(
                DS_LOAD_B128 " v[0:3], %0"
                :
                : "v"(ptr)
                : "v0", "v1", "v2", "v3"
            );
        } else if constexpr (sizeof(T) == 12) {
            asm volatile(
                DS_LOAD_B96 " v[0:2], %0"
                :
                : "v"(ptr)
                : "v0", "v1", "v2"
            );
        } else if constexpr (sizeof(T) == 8) {
            asm volatile(
                DS_LOAD_B64 " v[0:1], %0"
                :
                : "v"(ptr)
                : "v0", "v1"
            );
        } else if constexpr (sizeof(T) == 4) {
            asm volatile(
                DS_LOAD_B32 " v0, %0"
                :
                : "v"(ptr)
                : "v0"
            );
        }
    }

    // Ensure that all warps of the block finish at the same time
    __syncthreads();
}

template<typename T>
void run_tests(benchmark::executor& exec, int padding_words) {
    const auto grid_size = exec.dev.properties.compute_units * 512;
    const auto words_per_item = sizeof(T) / sizeof(word);
    const auto total_read = benchmark::size(words_per_thread * sizeof(word) * block_size * grid_size);
    const auto aligned = padding_words == 0 || padding_words % words_per_item == 0;

    const gpu::launch_config cfg = {
        .grid_size = grid_size,
        .block_size = block_size,
        // ensure one block per CU, just use the whole all of it
        .shared_mem_per_block = 65536,
    };

    const auto stats = exec.bench([&](const auto& stream) {
        stream.launch(cfg, kernel<T>, padding_words);
    });

    std::cout
        << "access: " << words_per_item << " words, padding: " << padding_words << " words, "
        << "aligned: " << (aligned ? "Y" : "N") << ": "
        << std::fixed << std::setprecision(2)
        << stats.runtime.average
        << "\t" << benchmark::throughput(total_read, stats.runtime.average).giga() << " GB/s"
        << std::endl;
}

int main() {
    try {
        const auto dev = gpu::get_default_device();
        auto exec = benchmark::executor(dev);

        std::cout << "block_size: " << block_size << std::endl;

        const auto run = [&](int padding_words) {
            run_tests<uint32_t>(exec, padding_words);
            run_tests<uint64_t>(exec, padding_words);
            run_tests<u96>(exec, padding_words);
            run_tests<u128>(exec, padding_words);
        };

        run(0);
        run(1);
        run(2);
        run(3);
        run(4);
    } catch (const gpu::error& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::cerr << e.trace << "\n";
        std::exit(1);
    } catch (const std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::exit(1);
    }
}

