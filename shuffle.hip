#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>

#include "gpu.hpp"
#include "benchmark.hpp"

constexpr int trails_per_thread = 128;

enum class shuffle_method {
    mov,
    ds_permute,
    ds_bpermute,
    ds_swizzle,
    v_readlane,
    v_writelane,
    v_readfirstlane,
    dpp,
};

template <int block_size, typename F>
__global__ __launch_bounds__(block_size)
void test_kernel(F f) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < trails_per_thread; ++i) {
        f();
    }

    asm volatile("s_waitcnt lgkmcnt(0)");
}

template<typename F>
void test(benchmark::executor& exec, const char* name, F f) {
    constexpr auto block_size = 1024;
    const auto grid_size = 4096 * exec.dev.properties.compute_units;
    const auto warp_size = exec.dev.properties.warp_size;
    const auto size = benchmark::size(warp_size * trails_per_thread * block_size * grid_size);
    const auto size_bytes = size.to_bytes<int>();

    const gpu::launch_config cfg = {
        .grid_size = grid_size,
        .block_size = block_size,
    };

    const auto stats = exec.bench([&](const auto& stream) {
        stream.launch(cfg, test_kernel<block_size, F>, f);
    });

    std::cout << name << ":\n";
    std::cout << "  time per launch: " << std::chrono::duration_cast<std::chrono::microseconds>(stats.average)
        << " +- " << std::chrono::duration_cast<std::chrono::microseconds>(stats.stddev) << "\n";
    std::cout << "  throughput:      " << benchmark::throughput(size, stats.average).tera() << " TOPS ("
       << benchmark::throughput(size_bytes, stats.average).tera() << " TB/s)\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(2);

    try {
        const auto dev = gpu::get_default_device();
        auto exec = benchmark::executor(dev);

        test(exec, "mov", [] {
            asm volatile("v_mov_b32 v0, v1" ::: "v0", "v1");
        });
        test(exec, "ds_permute (src reg == dst reg)", [] {
            asm volatile("ds_permute_b32 v0, v0, v1" ::: "v0", "v1");
        });
        test(exec, "ds_permute (src reg != dst reg)", [] {
            asm volatile("ds_permute_b32 v0, v2, v1" ::: "v0", "v1", "v2");
        });
        test(exec, "ds_bpermute (src reg == dst reg)", [] {
            asm volatile("ds_permute_b32 v0, v0, v1" ::: "v0", "v1");
        });
        test(exec, "ds_bpermute (src reg != dst reg)", [] {
            asm volatile("ds_permute_b32 v0, v2, v1" ::: "v0", "v1", "v2");
        });
        test(exec, "ds_swizzle (bcast32, src reg == dst reg)", [] {
             // BCAST32, src lane = 0
            asm volatile("ds_swizzle_b32 v0, v0 offset:0" ::: "v0", "v1");
        });
        test(exec, "ds_swizzle (bcast32, src reg != dst reg)", [] {
             // BCAST32, src lane = 0
            asm volatile("ds_swizzle_b32 v0, v1 offset:0" ::: "v0", "v1");
        });
        test(exec, "v_readlane (constant src lane)", [] {
            asm volatile("v_readlane_b32 s0, v0, 10" ::: "s0", "v0");
        });
        test(exec, "v_readlane (dynamic src lane)", [] {
            asm volatile("v_readlane_b32 s0, v0, s1" ::: "s0", "s1", "v0");
        });
        test(exec, "v_writelane (constant dst lane, dynamic src)", [] {
            asm volatile("v_writelane_b32 v0, s0, 10" ::: "v0", "s0");
        });
        test(exec, "v_writelane (dynamic dst lane, constant src)", [] {
            asm volatile("v_writelane_b32 v0, 10, s0" ::: "v0", "s0");
        });
        test(exec, "v_readfirstlane", [] {
            asm volatile("v_readfirstlane_b32 s0, v0" ::: "s0", "v0");
        });
        test(exec, "mov_dpp (row mirror, src reg == dst reg)", [] {
            // RDNA and CDNA both have this dpp instruction
            asm volatile("v_mov_b32_dpp v0, v0 row_mirror row_mask:0xf bank_mask:0xf" ::: "v0");
        });
        test(exec, "mov_dpp (row mirror, src reg != dst reg)", [] {
            // RDNA and CDNA both have this dpp instruction
            asm volatile("v_mov_b32_dpp v0, v1 row_mirror row_mask:0xf bank_mask:0xf" ::: "v0", "v1");
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
