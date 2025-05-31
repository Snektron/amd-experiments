#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>

#include "gpu.hpp"
#include "benchmark.hpp"

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
    const auto grid_size = 256 * exec.dev_props.multiProcessorCount;
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
        std::cout << "  time per launch: " << std::chrono::duration_cast<std::chrono::microseconds>(stats.average)
            << " +- " << std::chrono::duration_cast<std::chrono::microseconds>(stats.stddev) << "\n";
        std::cout << "  throughput:      " << benchmark::throughput(size, stats.average).tera() << " TOPS ("
            << benchmark::throughput(size_bytes, stats.average).tera() << " TB/s)\n";
        std::cout << "  cycles:          " << (double(exec.dev_props.clockRate) * 1000 * exec.dev_props.multiProcessorCount * 4 * exec.dev_props.warpSize) / benchmark::throughput(size, stats.fastest).rate << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(2);

    try {
        auto exec = benchmark::executor(gpu::default_device);

        const std::string arch_name{exec.dev_props.gcnArchName};

        // "Conflicts" here are not LDS bank conflicts but "collisions" when multiple lanes access
        // the same address. Even this is technically a bank conflict, ds_write/ds_read do not
        // suffer from it, this means that hardware uses some kind of broadcasting in this case.

        // uint32
        if (arch_name.find("gfx9") == 0 || arch_name.find("gfx10") == 0) {
            test<uint32_t>(exec, "ds_write_b32", [](auto addr, auto data) {
                #if !(defined(__GFX11__) || defined(__GFX12__))
                asm volatile("ds_write_b32 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "ds_read_b32", [](auto addr, auto data) {
                #if !(defined(__GFX11__) || defined(__GFX12__))
                uint32_t rtn;
                asm volatile("ds_read_b32 %0, %1" : "=&v"(rtn) : "v"(addr) : "memory");
                #endif
            });
        } else {
            test<uint32_t>(exec, "ds_store_b32", [](auto addr, auto data) {
                #if (defined(__GFX11__) || defined(__GFX12__))
                asm volatile("ds_store_b32 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint32_t>(exec, "ds_load_b32", [](auto addr, auto data) {
                #if (defined(__GFX11__) || defined(__GFX12__))
                uint32_t rtn;
                asm volatile("ds_load_b32 %0, %1" : "=&v"(rtn) : "v"(addr) : "memory");
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
        if (arch_name.find("gfx9") == 0 || arch_name.find("gfx10") == 0) {
            test<uint64_t>(exec, "ds_write_b64", [](auto addr, auto data) {
                #if !(defined(__GFX11__) || defined(__GFX12__))
                asm volatile("ds_write_b64 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "ds_read_b64", [](auto addr, auto data) {
                #if !(defined(__GFX11__) || defined(__GFX12__))
                uint64_t rtn;
                asm volatile("ds_read_b64 %0, %1" : "=&v"(rtn) : "v"(addr) : "memory");
                #endif
            });
        } else {
            test<uint64_t>(exec, "ds_store_b64", [](auto addr, auto data) {
                #if (defined(__GFX11__) || defined(__GFX12__))
                asm volatile("ds_store_b64 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<uint64_t>(exec, "ds_load_b64", [](auto addr, auto data) {
                #if (defined(__GFX11__) || defined(__GFX12__))
                uint64_t rtn;
                asm volatile("ds_load_b64 %0, %1" : "=&v"(rtn) : "v"(addr) : "memory");
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
                #if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__)
                asm volatile("ds_add_f64 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<double>(exec, "ds_add_rtn_f64", [](auto addr, auto data) {
                #if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__)
                double rtn;
                asm volatile("ds_add_rtn_f64 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<double>(exec, "ds_max_f64", [](auto addr, auto data) {
                #if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__)
                asm volatile("ds_max_f64 %0, %1" :: "v"(addr), "v"(data) : "memory");
                #endif
            });
            test<double>(exec, "ds_max_rtn_f64", [](auto addr, auto data) {
                #if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__)
                double rtn;
                asm volatile("ds_max_rtn_f64 %0, %1, %2" : "=&v"(rtn) : "v"(addr), "v"(data) : "memory");
                #endif
            });
        }

        // TODO: gfx12 and gfx942 also have packed 16-bit floats:
        // ds_pk_add_f16
        // ds_pk_add_bf16
        // ds_pk_add_rtn_f16
        // ds_pk_add_rtn_bf16
    } catch (const gpu::error& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::cerr << e.trace << "\n";
        std::exit(1);
    } catch (const std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::exit(1);
    }
}
