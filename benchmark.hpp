#ifndef _BENCHMARK_HPP
#define _BENCHMARK_HPP

#include <chrono>
#include <vector>
#include <ranges>
#include <algorithm>
#include <thread>

#include "gpu.hpp"

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
            return static_cast<double>(this->count) / 1000'000'000.f;
        }
    };

    struct throughput {
        using double_seconds = std::chrono::duration<double>;

        double rate;

        constexpr throughput(size z, duration time):
            rate(z.count / std::chrono::duration_cast<double_seconds>(time).count())
        { }

        constexpr double giga() const {
            return this->rate / 1000'000'000.f;
        }
    };

    struct stats {
        duration average;
        duration stddev;
        duration fastest;
        duration slowest;
    };

    template <typename F>
    stats run(const gpu::device& dev, F f) {
        using ns = std::chrono::nanoseconds;

        const auto stream = dev.create_stream(gpu::stream::flags::non_blocking);

        const auto max_cache_size = dev.largest_cache_size();
        const auto cache_buffer = dev.alloc<std::byte>(max_cache_size);

        auto events = std::vector<std::pair<gpu::event, gpu::event>>(iterations);

        for (int i = 0; i < warmups; ++i) {
            stream.memset(cache_buffer.raw, 0x00, max_cache_size);
            dev.sync();
            f(stream);
            dev.sync();
        }

        for (const auto& [start, stop] : events) {
            stream.memset(cache_buffer.raw, 0x00, max_cache_size);
            dev.sync();
            stream.record(start);
            f(stream);
            stream.record(stop);
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
}

#endif
