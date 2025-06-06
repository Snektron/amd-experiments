#ifndef _BENCHMARK_HPP
#define _BENCHMARK_HPP

#include "gpu.hpp"
#include "common.hpp"

#include <rocm_smi/rocm_smi.h>

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

    struct stats {
        duration average;
        duration stddev;
        duration fastest;
        duration slowest;
    };

    struct rsmi_error: traced_error {
        rsmi_status_t status;

        static const char* strerror(rsmi_status_t status) {
            const char* errstr;
            if (rsmi_status_string(status, &errstr) != RSMI_STATUS_SUCCESS) {
                return "(unknown)";
            }
            return errstr;
        }

        explicit rsmi_error(rsmi_status_t status):
            traced_error("{} ({})", strerror(status), static_cast<int>(status)),
            status(status)
        {
            assert(status != RSMI_STATUS_SUCCESS);
        }
    };

    #define RSMI_TRY(expr) {                        \
        const auto _result = (expr);                \
        if (_result != RSMI_STATUS_SUCCESS) {       \
            throw ::benchmark::rsmi_error(_result); \
        }                                           \
    }

    struct executor {
        const gpu::device& dev;
        gpu::stream stream;
        size_t max_cache_size;
        gpu::ptr<std::byte> cache_buffer;
        uint32_t rsmi_dev;

        rsmi_dev_perf_level_t orig_perf_level = RSMI_DEV_PERF_LEVEL_UNKNOWN;

        explicit executor(const gpu::device& dev):
            dev(dev),
            stream(this->dev.create_stream(gpu::stream::flags::non_blocking)),
            max_cache_size(this->dev.largest_cache_size()),
            cache_buffer(this->dev.alloc<std::byte>(this->max_cache_size))
        {
            RSMI_TRY(rsmi_init(0));

            // Try to fetch the RSMI device ID from the HIP device
            // ID. This is relatively janky, see
            // https://github.com/ROCm/rocm_smi_lib/issues/122#issuecomment-1839991753 and
            // https://github.com/ROCm/ROCmValidationSuite/blob/eaaaa4e093041a76c6367509dc04b2de2fbf67e2/src/gpu_util.cpp#L436

            this->rsmi_dev = [&]{
                const auto hip_pci_id = this->dev.properties.pci_address.rsmi_id();

                uint32_t num_devices;
                RSMI_TRY(rsmi_num_monitor_devices(&num_devices));
                for (uint32_t i = 0; i < num_devices; ++i) {
                    uint64_t rsmi_pci_id;
                    RSMI_TRY(rsmi_dev_pci_id_get(i, &rsmi_pci_id));

                    if (rsmi_pci_id == hip_pci_id) {
                        return i;
                    }
                }

                throw traced_error("could not map HIP device id {} to an RSMI device id", this->dev.hip_ordinal);
            }();

            char dev_name[256] = {0};
            RSMI_TRY(rsmi_dev_name_get(this->rsmi_dev, dev_name, sizeof(dev_name) - 1));
            std::cout << std::format("benchmarking on device '{}' ({})\n", this->dev.properties.device_name, this->dev.properties.pci_address);

            // Try to make performance deterministic
            // First query the current level so that we can reset it later.
            auto status = rsmi_dev_perf_level_get(this->rsmi_dev, &this->orig_perf_level);
            if (status != RSMI_STATUS_SUCCESS) {
                std::cerr << "warning: failed to query current perf level: " << rsmi_error::strerror(status) << "\n";
            }

            // "Determinism" mode doesn't always work, so use stable peak instead.
            // TODO: Should we set the profile too?
            status = rsmi_dev_perf_level_set_v1(this->rsmi_dev, RSMI_DEV_PERF_LEVEL_STABLE_PEAK);
            if (status == RSMI_STATUS_PERMISSION) {
                std::cerr << "warning: could not set perf level: insufficient permissions\n";
            } else if (status != RSMI_STATUS_SUCCESS) {
                std::cerr << "warning: failed to set perf level: " << rsmi_error::strerror(status) << "\n";
            }
        }

        executor(const executor&) = delete;
        executor& operator=(const executor&) = delete;

        executor(executor&&) = delete;
        executor& operator=(executor&&) = delete;

        ~executor() {
            if (this->orig_perf_level != RSMI_DEV_PERF_LEVEL_UNKNOWN) {
                rsmi_dev_perf_level_t current_level;
                auto status = rsmi_dev_perf_level_get(this->rsmi_dev, &current_level);
                if (status == RSMI_STATUS_SUCCESS && current_level != this->orig_perf_level) {
                    status = rsmi_dev_perf_level_set_v1(this->rsmi_dev, this->orig_perf_level);
                    if (status != RSMI_STATUS_SUCCESS ) {
                        std::cerr << "warning: failed to reset current perf level: " << rsmi_error::strerror(status) << "\n";
                    }
                }
            }


            assert(rsmi_shut_down() == RSMI_STATUS_SUCCESS);
        }

        template <typename F>
        stats bench(F f) {
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
                .average = duration(avg),
                .stddev = duration(stddev),
                .fastest = fastest,
                .slowest = slowest,
            };
        }
    };
}

#endif
