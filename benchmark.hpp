#ifndef _BENCHMARK_HPP
#define _BENCHMARK_HPP

#include "gpu.hpp"
#include "common.hpp"

#include <amd_smi/amdsmi.h>

#include <chrono>
#include <vector>
#include <ranges>
#include <algorithm>
#include <thread>

namespace benchmark {
    using duration = std::chrono::duration<double, std::nano>;

    constexpr size_t warmups = 10;
    constexpr size_t iterations = 50;

    struct size {
        size_t count;

        explicit constexpr size(size_t count): count(count) {}

        template <typename T>
        constexpr size to_bytes() const {
            return size(this->count * sizeof(T));
        }

        constexpr double giga() const {
            return static_cast<double>(this->count) / 1'000'000'000.f;
        }

        constexpr double tera() const {
            return static_cast<double>(this->count) / 1'000'000'000'000.f;
        }
    };

    struct throughput {
        using double_seconds = std::chrono::duration<double>;

        double rate;

        constexpr throughput(size z, duration time):
            rate(z.count / std::chrono::duration_cast<double_seconds>(time).count())
        { }

        constexpr double giga() const {
            return this->rate / 1'000'000'000.f;
        }

        constexpr double tera() const {
            return this->rate / 1'000'000'000'000.f;
        }
    };

    struct amdsmi_error: traced_error {
        amdsmi_status_t status;

        static const char* strerror(amdsmi_status_t status) {
            const char* errstr;
            if (amdsmi_status_code_to_string(status, &errstr) != AMDSMI_STATUS_SUCCESS) {
                return "(unknown)";
            }
            return errstr;
        }

        explicit amdsmi_error(amdsmi_status_t status):
            traced_error("{} ({})", strerror(status), static_cast<int>(status)),
            status(status)
        {
            assert(status != AMDSMI_STATUS_SUCCESS);
        }
    };

    #define AMDSMI_TRY(expr) {                        \
        const auto _result = (expr);                  \
        if (_result != AMDSMI_STATUS_SUCCESS) {       \
            throw ::benchmark::amdsmi_error(_result); \
        }                                             \
    }

    template<typename T>
    struct stats {
        T average;
        T stddev;
        T fastest;
        T slowest;
    };

    struct benchmark_stats {
        stats<duration> runtime;
    };

    struct executor {
        const gpu::device& dev;
        gpu::stream stream;
        size_t max_cache_size;
        gpu::ptr<std::byte> cache_buffer;
        amdsmi_processor_handle amdsmi_dev;

        amdsmi_dev_perf_level_t orig_perf_level = AMDSMI_DEV_PERF_LEVEL_UNKNOWN;

        explicit executor(const gpu::device& dev):
            dev(dev),
            stream(this->dev.create_stream(gpu::stream::flags::non_blocking)),
            max_cache_size(this->dev.largest_cache_size()),
            cache_buffer(this->dev.alloc<std::byte>(this->max_cache_size))
        {
            AMDSMI_TRY(amdsmi_init(AMDSMI_INIT_AMD_GPUS));

            amdsmi_processor_handle amdsmi_dev;
            const auto addr = amdsmi_bdf_t{
                .function_number = dev.properties.pci_address.function,
                .device_number = dev.properties.pci_address.device,
                .bus_number = dev.properties.pci_address.bus,
                .domain_number = dev.properties.pci_address.domain,
            };
            AMDSMI_TRY(amdsmi_get_processor_handle_from_bdf(addr, &this->amdsmi_dev));

            std::cout << std::format("benchmarking on device '{}' ({})\n", this->dev.properties.device_name, this->dev.properties.pci_address);

            // Try to make performance deterministic
            // First query the current level so that we can reset it later.
            auto status = amdsmi_get_gpu_perf_level(this->amdsmi_dev, &this->orig_perf_level);
            if (status != AMDSMI_STATUS_SUCCESS) {
                std::cerr << "warning: failed to query current perf level: " << amdsmi_error::strerror(status) << "\n";
            }

            // "Determinism" mode doesn't always work, so use stable peak instead.
            // TODO: Should we set the profile too?
            status = amdsmi_set_gpu_perf_level(this->amdsmi_dev, AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK);
            if (status == AMDSMI_STATUS_NO_PERM) {
                std::cerr << "warning: could not set perf level: insufficient permissions\n";
            } else if (status != AMDSMI_STATUS_SUCCESS) {
                std::cerr << "warning: failed to set perf level: " << amdsmi_error::strerror(status) << "\n";
            }
        }

        executor(const executor&) = delete;
        executor& operator=(const executor&) = delete;

        executor(executor&&) = delete;
        executor& operator=(executor&&) = delete;

        ~executor() {
            if (this->orig_perf_level != AMDSMI_DEV_PERF_LEVEL_UNKNOWN) {
                amdsmi_dev_perf_level_t current_level;
                auto status = amdsmi_get_gpu_perf_level(this->amdsmi_dev, &current_level);
                if (status == AMDSMI_STATUS_SUCCESS && current_level != this->orig_perf_level) {
                    status = amdsmi_set_gpu_perf_level(this->amdsmi_dev, this->orig_perf_level);
                    if (status != AMDSMI_STATUS_SUCCESS ) {
                        std::cerr << "warning: failed to reset current perf level: " << amdsmi_error::strerror(status) << "\n";
                    }
                }
            }

            assert(amdsmi_shut_down() == AMDSMI_STATUS_SUCCESS);
        }

        uint64_t get_gpu_sclk_freq_mhz() const {
            amdsmi_frequencies_t freqs;
            AMDSMI_TRY(amdsmi_get_clk_freq(
                this->amdsmi_dev,
                AMDSMI_CLK_TYPE_GFX,
                &freqs
            ));

            return freqs.frequency[freqs.current];
        }

        template <typename F>
        benchmark_stats bench(F f) {
            using ns = std::chrono::nanoseconds;

            auto events = std::vector<std::pair<gpu::event, gpu::event>>(iterations);

            for (int i = 0; i < warmups; ++i) {
                this->stream.memset(this->cache_buffer.raw, 0x00, this->max_cache_size);
                dev.sync();
                f(this->stream);
                dev.sync();
            }

            for (const auto& [start, stop] : events) {
                this->stream.memset(this->cache_buffer.raw, 0x00, this->max_cache_size);
                dev.sync();
                this->stream.record(start);
                f(this->stream);
                this->stream.record(stop);
                dev.sync();
            }

            auto durations = std::vector<duration>();
            durations.reserve(iterations);
            for (const auto& [start, stop] : events) {
                const auto elapsed = std::chrono::duration_cast<duration>(gpu::event::elapsed(start, stop));
                durations.push_back(elapsed);
            }

            const auto [fastest, slowest] = std::ranges::minmax(durations);
            const auto total = std::ranges::fold_left(durations, duration{0}, std::plus<>{});
            const auto avg = total.count() / iterations;
            const auto stddev = std::sqrt(
                std::ranges::fold_left(
                    durations
                        | std::views::transform([&](const auto time) {
                            const auto diff = time.count() - avg;
                            return diff * diff;
                        }),
                    0.0,
                    std::plus<>{}
                ) / iterations
            );

            return {
                .runtime = {
                    .average = duration(avg),
                    .stddev = duration(stddev),
                    .fastest = fastest,
                    .slowest = slowest,
                },
            };
        }
    };
}

#endif
