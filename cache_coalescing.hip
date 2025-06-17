#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <ranges>
#include <stacktrace>
#include <bit>

#include "gpu.hpp"
#include "benchmark.hpp"

constexpr int block_size = 1024;

// This value is used to permute accesses within a wavefront of 16 * warp_size bytes.
// It should be co-prime to the wave size to ensure a proper permutation and no duplicate
// lanes.
// Also, we're using a value thats larger than the cache line size, as we are interested
// in the worst-case situation. When the scramble range is smaller than this, it will just
// be equivalent to scramble_factor % scramble_range. This is still co-prime to the
// scramble_factor, as the scramble_range should always be a power of 2 and so only
// have divisors which are a power of 2.
constexpr int scramble_factor = 131111;

using u128 = __uint128_t;

template<typename T, bool enable_cache>
__global__ __launch_bounds__(block_size)
void kernel(T* __restrict__ buffer, int total_bits, int scramble_bits) {
    const auto warp_size = warpSize;

    const auto items_per_block = block_size;
    const auto items_per_warp = warp_size;

    const auto bid = blockIdx.x;
    const auto tid = threadIdx.x;

    const auto scramble_low_mask = (1 << scramble_bits) - 1;

    const auto index = bid * items_per_block + tid;
    const auto lo_addr = index & scramble_low_mask;
    const auto hi_addr = index & ~scramble_low_mask;
    const auto scrambled_index = hi_addr | ((lo_addr * scramble_factor) & scramble_low_mask);

    if constexpr (enable_cache) {
        gpu::do_not_optimize(buffer[scrambled_index]);
    } else {
        gpu::do_not_optimize(__builtin_nontemporal_load(&buffer[scrambled_index]));
    }
}

template <typename T, bool enable_cache>
void run_test(
    benchmark::executor& exec,
    size_t grid_size,
    benchmark::size read_bytes,
    const gpu::ptr<T>& buffer,
    size_t buffer_items,
    unsigned int scramble_range
) {
    assert((buffer_items & (buffer_items - 1)) == 0); // Must be a power of 2
    assert((scramble_range & (scramble_range - 1)) == 0); // Must be a power of 2

    const gpu::launch_config cfg = {
        .grid_size = grid_size,
        .block_size = block_size,
    };

    const auto stats = exec.bench([&](const auto& stream) {
        stream.launch(
            cfg,
            kernel<T, enable_cache>,
            buffer.raw,
            static_cast<int>(std::bit_width(buffer_items)),
            static_cast<int>(std::bit_width(scramble_range))
        );
    });

    std::cout << sizeof(T) << " * " << scramble_range << " = " << (scramble_range * sizeof(T)) << " bytes ("
        << (enable_cache ? "cached" : "uncached") << "): "
        << benchmark::throughput(read_bytes, stats.runtime.average).giga() << " GB/s"
        << std::endl;
}

template<typename T, bool enable_cache>
void run_tests(benchmark::executor& exec) {
    const auto grid_size = 512 * 1024;
    const auto buffer_items = grid_size * block_size;
    const auto read_bytes = benchmark::size(grid_size * block_size).to_bytes<T>();

    const auto buffer = exec.dev.alloc<T>(buffer_items);

    auto test = [&](int scramble_range) {
        run_test<T, enable_cache>(exec, grid_size, read_bytes, buffer, buffer_items, scramble_range);
    };

    test(1);
    test(16);
    test(32);
    test(64);
    for (int i = 128; i <= (1<<17); i <<= 1) {
        test(i);
    }
}

int main() {
    try {
        const auto dev = gpu::get_default_device();
        auto exec = benchmark::executor(dev);

        std::cout << "cache line size: " << dev.properties.cacheline_size << " B\n";
        std::cout << "device cache sizes:\n";
        for (int i = 0; i < dev.properties.cache_size.size(); ++i) {
            if (dev.properties.cache_size[i] != 0) {
                std::cout << "  l" << (i + 1) << ": " << (dev.properties.cache_size[i] / 1024) << " KB\n";
            }
        }
        std::cout << std::endl;

        run_tests<uint32_t, true>(exec);
        run_tests<u128, true>(exec);
        run_tests<uint32_t, false>(exec);
        run_tests<u128, false>(exec);
    } catch (const gpu::error& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::cerr << e.trace << "\n";
        std::exit(1);
    } catch (const std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::exit(1);
    }
}
