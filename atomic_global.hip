#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>

#include "gpu.hpp"
#include "benchmark.hpp"

#if defined(__GFX11__) || defined(__GFX12__)
#define USE_NEW_INSTRUCTION_NAMES 1
#else
#define USE_NEW_INSTRUCTION_NAMES 0
#endif

#if defined(__GFX12__)
#define RETURN_MODIFIER " th:TH_ATOMIC_RETURN"
#elif defined(__gfx942__)
#define RETURN_MODIFIER " sc0"
#else
#define RETURN_MODIFIER " glc"
#endif

#if defined(__GFX12__)
#define COHERENT_MODIFIER ""
#elif defined(__gfx942__)
#define COHERENT_MODIFIER " sc1"
#else
#define COHERENT_MODIFIER " glc"
#endif

#if defined(__GFX12__)
#define SCOPE_MODIFIER " scope:SCOPE_DEV"
#else
#define SCOPE_MODIFIER ""
#endif

#if defined(__gfx90a__) || defined(__gfx942__) || defined(__GFX11__) || defined(__GFX12__)
#define HAS_GLOBAL_ATOMIC_ADD_F32 1
#else
#define HAS_GLOBAL_ATOMIC_ADD_F32 0
#endif

#if defined(__GFX12__)
#define HAS_GLOBAL_ATOMIC_MIN_NUM_F32 1
#else
#define HAS_GLOBAL_ATOMIC_MIN_NUM_F32 0
#endif

#if defined(__gfx90a__) || defined(__gfx942__)
#define HAS_GLOBAL_ATOMIC_ADD_F64 1
#define HAS_GLOBAL_ATOMIC_MIN_F64 1
#else
#define HAS_GLOBAL_ATOMIC_ADD_F64 0
#define HAS_GLOBAL_ATOMIC_MIN_F64 0
#endif

#if defined(__gfx942__) || defined(__GFX12__)
#define HAS_GLOBAL_ATOMIC_PK_ADD 1
#else
#define HAS_GLOBAL_ATOMIC_PK_ADD 0
#endif

constexpr int trials_per_thread = 64;

template <typename T, int block_size, typename F>
__global__ __launch_bounds__(block_size)
void test_kernel(F f, int conflicts_shift, T* buffer) {
    __shared__ T shared[block_size];

    #pragma clang loop unroll_count(16)
    for (int i = 0; i < trials_per_thread; ++i) {
        auto* addr = &buffer[blockIdx.x * block_size + (threadIdx.x >> conflicts_shift)];
        f(addr, static_cast<T>(threadIdx.x));

        if ((i + 1) % 16 == 0) {
            #if defined(__GFX12__)
            asm volatile("s_wait_loadcnt 0x0");
            asm volatile("s_wait_storecnt 0x0");
            #else
            asm volatile("s_waitcnt vmcnt(0)");
            #endif
        }
    }
}

template<typename T, typename F>
void test(benchmark::executor& exec, const char* name, F f) {
    constexpr auto block_size = 256;
    const auto grid_size = 256 * exec.dev.properties.compute_units;
    const auto items = trials_per_thread * block_size * grid_size;
    const auto size = benchmark::size(items);
    const auto size_bytes = size.to_bytes<T>();

    const gpu::launch_config cfg = {
        .grid_size = grid_size,
        .block_size = block_size,
    };

    const auto buffer = exec.dev.alloc<T>(items);

    for (int conflicts_shift = 0; conflicts_shift < 6; ++conflicts_shift) {
        const auto stats = exec.bench([&](const auto& stream) {
            stream.launch(cfg, test_kernel<T, block_size, F>, f, conflicts_shift, buffer.raw);
        });

        std::cout << name << " with " << (1 << conflicts_shift) << " threads per address:\n";
        std::cout << "  time per launch: " << std::chrono::duration_cast<std::chrono::microseconds>(stats.average)
            << " +- " << std::chrono::duration_cast<std::chrono::microseconds>(stats.stddev) << "\n";
        std::cout << "  throughput:      " << benchmark::throughput(size, stats.average).tera() << " TOPS ("
            << benchmark::throughput(size_bytes, stats.average).tera() << " TB/s)\n";
    std::cout << "  cycles:          " << (double(exec.dev.properties.clock_rate) * 1000 * exec.dev.properties.total_simds() * exec.dev.properties.warp_size) / benchmark::throughput(size, stats.fastest).rate << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(2);

    try {
        const auto dev = gpu::get_default_device();
        auto exec = benchmark::executor(dev);

        const auto arch_name = exec.dev.properties.arch_name;

        // "Conflicts" here are not LDS bank conflicts but "collisions" when multiple lanes access
        // the same address. Even this is technically a bank conflict, ds_write/ds_read do not
        // suffer from it, this means that hardware uses some kind of broadcasting in this case.

        const bool use_new_instruction_names =
            arch_name.find("gfx11") == 0 || arch_name.find("gfx12") == 0;

        // uint32
        if (use_new_instruction_names) {
            test<uint32_t>(exec, "global_store_b32", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_store_b32 %0, %1, off" COHERENT_MODIFIER SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_load_b32", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                uint32_t rtn;
                asm volatile("global_load_b32 %0, %1, off" COHERENT_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_add_u32", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_add_u32 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_add_u32 return", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                uint32_t rtn;
                asm volatile("global_atomic_add_u32 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_min_u32", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_min_u32 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_min_u32 return", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                uint32_t rtn;
                asm volatile("global_atomic_min_u32 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_and_b32", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_and_b32 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_and_b32 return", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                uint32_t rtn;
                asm volatile("global_atomic_and_b32 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
        } else {
            test<uint32_t>(exec, "global_store_dword", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_store_dword %0, %1, off" COHERENT_MODIFIER SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_load_dword", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                uint32_t rtn;
                asm volatile("global_load_dword %0, %1, off" COHERENT_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_add", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_add %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_add return", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                uint32_t rtn;
                asm volatile("global_atomic_add %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_umax", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_umax %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_umax return", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                uint32_t rtn;
                asm volatile("global_atomic_umax %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_and", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_and %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_and return", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                uint32_t rtn;
                asm volatile("global_atomic_and %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
        }

        // uint64
        if (use_new_instruction_names) {
            test<uint64_t>(exec, "global_store_b64", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_store_b64 %0, %1, off" COHERENT_MODIFIER SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_load_b64", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                uint64_t rtn;
                asm volatile("global_load_b64 %0, %1, off" COHERENT_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_add_u64", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_add_u64 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_add_u64 return", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                uint64_t rtn;
                asm volatile("global_atomic_add_u64 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_min_u64", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_min_u64 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_min_u64 return", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                uint64_t rtn;
                asm volatile("global_atomic_min_u64 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_and_b64", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_and_b64 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_and_b64 return", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                uint64_t rtn;
                asm volatile("global_atomic_and_b64 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
        } else {
            test<uint64_t>(exec, "global_store_dwordx2", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_store_dwordx2 %0, %1, off" COHERENT_MODIFIER SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_load_dwordx2", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                uint64_t rtn;
                asm volatile("global_load_dwordx2 %0, %1, off" COHERENT_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_add_x2", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_add_x2 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_add_x2 return", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                uint64_t rtn;
                asm volatile("global_atomic_add_x2 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_umax_x2", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_umax_x2 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_umax_x2 return", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                uint64_t rtn;
                asm volatile("global_atomic_umax_x2 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_and_x2", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                asm volatile("global_atomic_and_x2 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "global_atomic_and_x2 return", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                uint64_t rtn;
                asm volatile("global_atomic_and_x2 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
        }

        // float
        if (arch_name.find("gfx90a") == 0 || arch_name.find("gfx942") == 0 || arch_name.find("gfx11") == 0 || arch_name.find("gfx12") == 0) {
            test<float>(exec, "global_atomic_add_f32", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_ADD_F32
                asm volatile("global_atomic_add_f32 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<float>(exec, "global_atomic_add_f32 return", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_ADD_F32
                float rtn;
                asm volatile("global_atomic_add_f32 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
        }
        if (arch_name.find("gfx12") == 0) {
            test<float>(exec, "global_atomic_min_num_f32", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_MIN_NUM_F32
                asm volatile("global_atomic_min_num_f32 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<float>(exec, "global_atomic_min_num_f32 return", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_MIN_NUM_F32
                float rtn;
                asm volatile("global_atomic_min_num_f32 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
        }

        // double
        if (arch_name.find("gfx90a") == 0 || arch_name.find("gfx942") == 0) {
            test<double>(exec, "global_atomic_add_f64", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_ADD_F64
                asm volatile("global_atomic_add_f64 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<double>(exec, "global_atomic_add_f64 return", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_ADD_F64
                double rtn;
                asm volatile("global_atomic_add_f64 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<double>(exec, "global_atomic_min_f64", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_ADD_F64
                asm volatile("global_atomic_min_f64 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<double>(exec, "global_atomic_min_f64 return", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_ADD_F64
                double rtn;
                asm volatile("global_atomic_min_f64 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
        }

        // packed f16/bf16
        if (arch_name.find("gfx942") == 0 || arch_name.find("gfx12") == 0) {
            test<uint32_t>(exec, "global_atomic_pk_add_f16", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_PK_ADD
                asm volatile("global_atomic_pk_add_f16 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_pk_add_f16 return", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_PK_ADD
                uint32_t rtn;
                asm volatile("global_atomic_pk_add_f16 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_pk_add_bf16", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_PK_ADD
                asm volatile("global_atomic_pk_add_bf16 %0, %1, off" SCOPE_MODIFIER :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "global_atomic_pk_add_bf16 return", [](auto addr, auto data) {
                #if HAS_GLOBAL_ATOMIC_PK_ADD
                uint32_t rtn;
                asm volatile("global_atomic_pk_add_bf16 %0, %1, %2, off" RETURN_MODIFIER SCOPE_MODIFIER : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
        }
    } catch (const gpu::error& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::cerr << e.trace << "\n";
        std::exit(1);
    } catch (const std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::exit(1);
    }
}
