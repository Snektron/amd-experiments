#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <hip/hip_bf16.h>

#include "gpu.hpp"
#include "benchmark.hpp"

constexpr int trials_per_thread = 256;

struct undef {
    template <typename T>
    __device__ __forceinline__
    operator T() const {
        T value;
        asm volatile("" : "=v"(value));
        return value;
    }
};

template <typename T>
__device__
void do_not_optimize(T value) {
    asm volatile ("" :: "v"(value));
}

using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
using u32x8 = uint32_t __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));
using f32x16 = float __attribute__((ext_vector_type(16)));
using f16x8 = _Float16 __attribute__((ext_vector_type(8)));
using f16x16 = _Float16 __attribute__((ext_vector_type(16)));
using u16x16 = uint16_t __attribute__((ext_vector_type(16)));

template <gpu::family_set families, int block_size, typename F>
__global__ __launch_bounds__(block_size)
void test_kernel(F f) {
    if constexpr (families.contains(gpu::get_device_family())) {
        #pragma clang loop unroll(full)
        for (int i = 0; i < trials_per_thread; ++i) {
            f();
        }
    }
}

template<gpu::family_set families = gpu::family_set::all, typename F>
void test(benchmark::executor& exec, const char* name, F f, int ops_per_inst) {
    std::cout << name << ":\n";

    if (!families.contains(exec.dev.get_family())) {
        std::cout << "  skipping (not supported on this arch)\n";
        return;
    }

    constexpr auto block_size = 1024;
    const auto warp_size = exec.dev.properties.warp_size;
    const auto total_simds = exec.dev.properties.total_simds();
    const auto warps = block_size / warp_size;
    const auto grid_size = 32 * exec.dev.properties.compute_units;
    const auto insts = benchmark::size(trials_per_thread * warps * grid_size);
    const auto flop = benchmark::size(insts.count * ops_per_inst);

    const gpu::launch_config cfg = {
        .grid_size = grid_size,
        .block_size = block_size,
    };

    const auto stats = exec.bench([&](const auto& stream) {
        stream.launch(cfg, test_kernel<families, block_size, F>, f);
    });

    const auto clock_rate = stats.clock_rate.average;
    const auto cycles = clock_rate * std::chrono::duration_cast<std::chrono::duration<double>>(stats.runtime.largest).count();
    const auto latency = cycles / (insts.count / total_simds);
    const auto ops = ops_per_inst * exec.dev.properties.simds_per_cu / latency;

    std::cout << "  time per launch: " << std::chrono::duration_cast<std::chrono::microseconds>(stats.runtime.average)
        << " +- " << std::chrono::duration_cast<std::chrono::microseconds>(stats.runtime.stddev) << "\n";
    std::cout << "  throughput:      " << benchmark::throughput(insts, stats.runtime.average).giga() << " Ginst/s\n";
    std::cout << "  throughput:      " << benchmark::throughput(flop, stats.runtime.average).tera() << " TOPS\n";
    std::cout << "  latency:         " << latency << " cycles/inst\n";
    std::cout << "  ops per cu:      " << ops << " ops/CU/cycle\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(2);
    try {
        const auto dev = gpu::get_default_device();
        const auto ws = dev.properties.warp_size;
        auto exec = benchmark::executor(dev);

        // Common instructions

        test(exec, "mov", [] {
            asm volatile("v_mov_b32 v0, v1" ::: "v0", "v1");
        }, ws);

        test(exec, "v_mul_f32", [] {
            asm volatile("v_mul_f32 v0, v1, v2" ::: "v0", "v1", "v2");
        }, ws);

        test(exec, "v_fma_f32", [] {
            asm volatile("v_fma_f32 v0, v1, v2, v3" ::: "v0", "v1", "v2", "v3");
        }, 2 * ws);

        test(exec, "v_pk_fma_f16", [] {
            asm volatile("v_pk_fma_f16 v0, v1, v2, v3" ::: "v0", "v1", "v2", "v3");
        }, 4 * ws);

        // RDNA 3 instructions

        test<gpu::family_set::rdna3>(exec, "v_wmma_f32_16x16x16_f16", [] {
            do_not_optimize(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(undef(), undef(), undef()));
        }, 16 * 16 * 16 * 2);

        test<gpu::family_set::rdna3>(exec, "v_wmma_f16_16x16x16_f16", [] {
            do_not_optimize(__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(undef(), undef(), undef(), 0));
        }, 16 * 16 * 16 * 2);

        test<gpu::family_set::rdna3>(exec, "v_wmma_i32_16x16x16_iu8", [] {
            do_not_optimize(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(0, undef(), 0, undef(), undef(), 0));
        }, 16 * 16 * 16 * 2);

        test<gpu::family_set::rdna3>(exec, "v_wmma_i32_16x16x16_iu4", [] {
            do_not_optimize(__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(0, undef(), 0, undef(), undef(), 0));
        }, 16 * 16 * 16 * 2);

        // RDNA 4 instructions

        test<gpu::family_set::rdna4>(exec, "v_wmma_f32_16x16x16_f16", [] {
            do_not_optimize(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(undef(), undef(), undef()));
        }, 16 * 16 * 16 * 2);

        test<gpu::family_set::rdna4>(exec, "v_wmma_f16_16x16x16_f16", [] {
            do_not_optimize(__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(undef(), undef(), undef()));
        }, 16 * 16 * 16 * 2);

        test<gpu::family_set::rdna4>(exec, "v_wmma_f32_16x16x16_fp8_fp8", [] {
            do_not_optimize(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(undef(), undef(), undef()));
        }, 16 * 16 * 16 * 2);

        test<gpu::family_set::rdna4>(exec, "v_wmma_i32_16x16x16_iu8", [] {
            do_not_optimize(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(0, undef(), 0, undef(), undef(), 0));
        }, 16 * 16 * 16 * 2);

        test<gpu::family_set::rdna4>(exec, "v_wmma_i32_16x16x16_iu4", [] {
            do_not_optimize(__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(0, undef(), 0, undef(), undef(), 0));
        }, 16 * 16 * 16 * 2);

        test<gpu::family_set::rdna4>(exec, "v_wmma_i32_16x16x32_iu4", [] {
            do_not_optimize(__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(0, undef(), 0, undef(), undef(), 0));
        }, 16 * 16 * 32 * 2);

        // CDNA 3 instructions

        test<gpu::family_set::cdna3>(exec, "v_mfma_f32_32x32x1_f32", [] {
            do_not_optimize(__builtin_amdgcn_mfma_f32_32x32x1f32(undef(), undef(), undef(), 0, 0, 0));
        }, 32 * 32 * 1 * 2);

        test<gpu::family_set::cdna3>(exec, "v_mfma_f32_16x16x1_f32", [] {
            do_not_optimize(__builtin_amdgcn_mfma_f32_16x16x1f32(undef(), undef(), undef(), 0, 0, 0));
        }, 16 * 16 * 1 * 2);

        test<gpu::family_set::cdna3>(exec, "v_mfma_f32_32x32x4_f16", [] {
            do_not_optimize(__builtin_amdgcn_mfma_f32_32x32x4f16(undef(), undef(), undef(), 0, 0, 0));
        }, 32 * 32 * 4 * 2);

        test<gpu::family_set::cdna3>(exec, "v_mfma_i32_16x16x4_i8", [] {
            do_not_optimize(__builtin_amdgcn_mfma_i32_16x16x4i8(undef(), undef(), undef(), 0, 0, 0));
        }, 16 * 16 * 4 * 2);

        test<gpu::family_set::cdna3>(exec, "v_mfma_f64_16x16x4_f64", [] {
            do_not_optimize(__builtin_amdgcn_mfma_f64_16x16x4f64(undef(), undef(), undef(), 0, 0, 0));
        }, 16 * 16 * 4 * 2);

        test<gpu::family_set::cdna3>(exec, "v_mfma_f64_4x4x4_f64", [] {
            do_not_optimize(__builtin_amdgcn_mfma_f64_4x4x4f64(undef(), undef(), undef(), 0, 0, 0));
        }, 4 * 4 * 4 * 2);

        test<gpu::family_set::cdna3>(exec, "v_mfma_i32_16x16x32_i8", [] {
            do_not_optimize(__builtin_amdgcn_mfma_i32_16x16x32_i8(undef(), undef(), undef(), 0, 0, 0));
        }, 16 * 16 * 32 * 2);

        test<gpu::family_set::cdna3>(exec, "v_mfma_i32_32x32x16_i8", [] {
            do_not_optimize(__builtin_amdgcn_mfma_i32_32x32x16_i8(undef(), undef(), undef(), 0, 0, 0));
        }, 32 * 32 * 16 * 2);

        test<gpu::family_set::cdna3>(exec, "v_mfma_f32_16x16x32_fp8_fp8", [] {
            do_not_optimize(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(undef(), undef(), undef(), 0, 0, 0));
        }, 16 * 16 * 32 * 2);

        test<gpu::family_set::cdna3>(exec, "v_mfma_f32_32x32x16_fp8_fp8", [] {
            do_not_optimize(__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(undef(), undef(), undef(), 0, 0, 0));
        }, 32 * 32 * 16 * 2);

    } catch (const gpu::error& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::cerr << e.trace << "\n";
        std::exit(1);
    } catch (const std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::exit(1);
    }
}
