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

#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__)
#define HAS_DS_F64_ATOMICS 1
#else
#define HAS_DS_F64_ATOMICS 0
#endif

#if defined(__gfx942__) || defined(__GFX12__)
#define HAS_DS_PK_ATOMICS 1
#else
#define HAS_DS_PK_ATOMICS 0
#endif

constexpr int trials_per_thread = 256;

template <typename T, int block_size, typename F>
__global__ __launch_bounds__(block_size)
void test_kernel(F f, int conflicts_shift) {
    __shared__ T shared[block_size];

    #pragma clang loop unroll_count(16)
    for (int i = 0; i < trials_per_thread; ++i) {
        auto* addr = (__attribute__((address_space(3))) T*)&shared[threadIdx.x >> conflicts_shift];
        f(addr, static_cast<T>(threadIdx.x));

        if ((i + 1) % 16 == 0) {
            #if defined(__GFX12__)
            asm volatile("s_wait_dscnt 0x0");
            #else
            asm volatile("s_waitcnt lgkmcnt(0)");
            #endif
        }
    }
}

template<typename T, typename F>
void test(benchmark::executor& exec, const char* name, F f) {
    constexpr auto block_size = 256;
    const auto grid_size = 256 * exec.dev.properties.compute_units;
    const auto size = benchmark::size(trials_per_thread * block_size * grid_size);
    const auto size_bytes = size.to_bytes<T>();

    const gpu::launch_config cfg = {
        .grid_size = grid_size,
        .block_size = block_size,
    };

    for (int conflicts_shift = 0; conflicts_shift < 6; ++conflicts_shift) {
        const auto stats = exec.bench([&](const auto& stream) {
            stream.launch(cfg, test_kernel<T, block_size, F>, f, conflicts_shift);
        });

        std::cout << name << " with " << (1 << conflicts_shift) << " threads per address:\n";
        std::cout << "  time per launch: " << std::chrono::duration_cast<std::chrono::microseconds>(stats.runtime.average)
            << " +- " << std::chrono::duration_cast<std::chrono::microseconds>(stats.runtime.stddev) << "\n";
        std::cout << "  throughput:      " << benchmark::throughput(size, stats.runtime.average).tera() << " TOPS ("
            << benchmark::throughput(size_bytes, stats.runtime.average).tera() << " TB/s)\n";
    std::cout << "  cycles:          " << (stats.clock_rate.average * exec.dev.properties.total_simds() * exec.dev.properties.warp_size) / benchmark::throughput(size, stats.runtime.smallest).rate << "\n";
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
            test<uint32_t>(exec, "ds_store_b32", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                asm volatile("ds_store_b32 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "ds_load_b32", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                uint32_t rtn;
                asm volatile("ds_load_b32 %0, %1" : "=&v"(rtn) : "v"(addr) : "memory");
                #endif
            });
        } else {
            test<uint32_t>(exec, "ds_write_b32", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                asm volatile("ds_write_b32 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "ds_read_b32", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                uint32_t rtn;
                asm volatile("ds_read_b32 %0, %1" : "=&v"(rtn) : "v"(addr) : "memory");
                #endif
            });
        }
        test<uint32_t>(exec, "ds_add_u32", [](auto addr, auto data) {
            asm volatile("ds_add_u32 %0, %1" :: "v"(addr), "v"(data) : "memory");
        });
        test<uint32_t>(exec, "ds_add_rtn_u32", [](auto addr, auto data) {
            uint32_t rtn;
            asm volatile("ds_add_rtn_u32 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
        });
        test<uint32_t>(exec, "ds_max_u32", [](auto addr, auto data) {
            asm volatile("ds_max_u32 %0, %1" :: "v"(addr), "v"(data) : "memory");
        });
        test<uint32_t>(exec, "ds_max_rtn_u32", [](auto addr, auto data) {
            uint32_t rtn;
            asm volatile("ds_max_rtn_u32 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
        });
        test<uint32_t>(exec, "ds_and_b32", [](auto addr, auto data) {
            asm volatile("ds_and_b32 %0, %1" :: "v"(addr), "v"(data) : "memory");
        });
        test<uint32_t>(exec, "ds_and_rtn_b32", [](auto addr, auto data) {
            uint32_t rtn;
            asm volatile("ds_and_rtn_b32 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
        });

        // uint64
        if (use_new_instruction_names) {
            test<uint64_t>(exec, "ds_store_b64", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                asm volatile("ds_store_b64 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "ds_load_b64", [](auto addr, auto data) {
                #if USE_NEW_INSTRUCTION_NAMES
                uint64_t rtn;
                asm volatile("ds_load_b64 %0, %1" : "=&v"(rtn) : "v"(addr) : "memory");
                #endif
            });
        } else {
            test<uint64_t>(exec, "ds_write_b64", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                asm volatile("ds_write_b64 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "ds_read_b64", [](auto addr, auto data) {
                #if !USE_NEW_INSTRUCTION_NAMES
                uint64_t rtn;
                asm volatile("ds_read_b64 %0, %1" : "=&v"(rtn) : "v"(addr) : "memory");
                #endif
            });
        }
        test<uint64_t>(exec, "ds_add_u64", [](auto addr, auto data) {
            asm volatile("ds_add_u64 %0, %1" :: "v"(addr), "v"(data) : "memory");
        });
        test<uint64_t>(exec, "ds_add_rtn_u64", [](auto addr, auto data) {
            uint64_t rtn;
            asm volatile("ds_add_rtn_u64 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
        });
        test<uint64_t>(exec, "ds_max_u64", [](auto addr, auto data) {
            asm volatile("ds_max_u64 %0, %1" :: "v"(addr), "v"(data) : "memory");
        });
        test<uint64_t>(exec, "ds_max_rtn_u64", [](auto addr, auto data) {
            uint64_t rtn;
            asm volatile("ds_max_rtn_u64 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
        });
        test<uint64_t>(exec, "ds_and_b64", [](auto addr, auto data) {
            asm volatile("ds_and_b64 %0, %1" :: "v"(addr), "v"(data) : "memory");
        });
        test<uint64_t>(exec, "ds_and_rtn_b64", [](auto addr, auto data) {
            uint64_t rtn;
            asm volatile("ds_and_rtn_b64 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
        });

        // float
        test<float>(exec, "ds_add_f32", [](auto addr, auto data) {
            asm volatile("ds_add_f32 %0, %1" :: "v"(addr), "v"(data) : "memory");
        });
        test<float>(exec, "ds_add_rtn_f32", [](auto addr, auto data) {
            float rtn;
            asm volatile("ds_add_rtn_f32 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
        });
        test<float>(exec, "ds_max_f32", [](auto addr, auto data) {
            asm volatile("ds_max_f32 %0, %1" :: "v"(addr), "v"(data) : "memory");
        });
        test<float>(exec, "ds_max_rtn_f32", [](auto addr, auto data) {
            float rtn;
            asm volatile("ds_max_rtn_f32 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
        });

        // double
        if (arch_name.find("gfx908") == 0 || arch_name.find("gfx90a") == 0 || arch_name.find("gfx942") == 0) {
            test<double>(exec, "ds_add_f64", [](auto addr, auto data) {
                #if HAS_DS_F64_ATOMICS
                asm volatile("ds_add_f64 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<double>(exec, "ds_add_rtn_f64", [](auto addr, auto data) {
                #if HAS_DS_F64_ATOMICS
                double rtn;
                asm volatile("ds_add_rtn_f64 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<double>(exec, "ds_max_f64", [](auto addr, auto data) {
                #if HAS_DS_F64_ATOMICS
                asm volatile("ds_max_f64 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<double>(exec, "ds_max_rtn_f64", [](auto addr, auto data) {
                #if HAS_DS_F64_ATOMICS
                double rtn;
                asm volatile("ds_max_rtn_f64 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
        }

        // packed f16/bf16
        if (arch_name.find("gfx942") == 0 || arch_name.find("gfx12") == 0) {
            test<uint32_t>(exec, "ds_pk_add_f16", [](auto addr, auto data) {
                #if HAS_DS_PK_ATOMICS
                asm volatile("ds_pk_add_f16 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "ds_pk_add_rtn_f16", [](auto addr, auto data) {
                #if HAS_DS_PK_ATOMICS
                uint32_t rtn;
                asm volatile("ds_pk_add_rtn_f16 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "ds_pk_add_bf16", [](auto addr, auto data) {
                #if HAS_DS_PK_ATOMICS
                asm volatile("ds_pk_add_bf16 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "ds_pk_add_rtn_bf16", [](auto addr, auto data) {
                #if HAS_DS_PK_ATOMICS
                uint32_t rtn;
                asm volatile("ds_pk_add_rtn_bf16 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
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
