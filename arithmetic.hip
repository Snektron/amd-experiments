#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>

#include "gpu.hpp"
#include "benchmark.hpp"

constexpr int trials_per_thread = 256;

template <int block_size, typename F>
__global__ __launch_bounds__(block_size)
void test_kernel(F f) {
    #pragma clang loop unroll_count(16)
    for (int i = 0; i < trials_per_thread; ++i) {
        f();
    }
}

template<typename F>
void test(benchmark::executor& exec, const char* name, F f) {
    constexpr auto block_size = 256;
    const auto grid_size = 1024 * exec.dev.properties.compute_units;
    const auto size = benchmark::size(trials_per_thread * block_size * grid_size);
    const auto size_bytes = size.to_bytes<int>();

    const gpu::launch_config cfg = {
        .grid_size = grid_size,
        .block_size = block_size,
    };

    const auto stats = exec.bench([&](const auto& stream) {
        stream.launch(cfg, test_kernel<block_size, F>, f);
    });

    std::cout << name << ":\n";
    std::cout << "  time per launch: " << std::chrono::duration_cast<std::chrono::microseconds>(stats.runtime.average)
        << " +- " << std::chrono::duration_cast<std::chrono::microseconds>(stats.runtime.stddev) << "\n";
    std::cout << "  throughput:      " << benchmark::throughput(size, stats.runtime.average).tera() << " TOPS ("
       << benchmark::throughput(size_bytes, stats.runtime.average).tera() << " TB/s)\n";
    std::cout << "  cycles:          " << (stats.clock_rate.average * exec.dev.properties.total_simds() * exec.dev.properties.warp_size) / benchmark::throughput(size, stats.runtime.smallest).rate << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(2);

    try {
        const auto dev = gpu::get_default_device();
        auto exec = benchmark::executor(dev);
        test(exec, "mov", [] {
            asm volatile("v_mov_b32 v0, v1" ::: "v0", "v1");
        });
        test(exec, "v_mul_u32_u24", [] {
            asm volatile("v_mul_u32_u24 v0, v1, v2" ::: "v0", "v1", "v2");
        });
        test(exec, "v_mul_hi_u32", [] {
            asm volatile("v_mul_hi_u32 v0, v1, v2" ::: "v0", "v1", "v2");
        });
        test(exec, "v_mul_lo_u32", [] {
            asm volatile("v_mul_lo_u32 v0, v1, v2" ::: "v0", "v1", "v2");
        });
        test(exec, "v_mad_u64_u32", [] {
            #if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
            asm volatile("v_mad_u64_u32 v[0:1], s0, v2, v3, v[4:5]" ::: "v0", "v1", "s0", "v2", "v3", "v4", "v5");
            #else
            asm volatile("v_mad_u64_u32 v[0:1], s[0:1], v2, v3, v[4:5]" ::: "v0", "v1", "s0", "s1", "v2", "v3", "v4", "v5");
            #endif
        });
    } catch (const gpu::error& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::cerr << e.trace << "\n";
        std::exit(1);
    } catch (const std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::exit(1);
    }
}
