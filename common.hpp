#ifndef _COMMON_H
#define _COMMON_H

#include <stacktrace>
#include <sstream>
#include <format>
#include <string>
#include <cstdint>
#include <vector>

struct traced_error: std::runtime_error {
    std::stacktrace trace;

    traced_error(const std::string& msg):
        runtime_error(msg),
        trace(std::stacktrace::current())
    {}

    template<typename... Args>
    traced_error(std::format_string<Args...> fmt, Args&&... args):
        runtime_error(std::format(fmt, std::forward<Args>(args)...)),
        trace(std::stacktrace::current())
    {}
};

struct pci_address {
    uint16_t domain;
    uint8_t bus;
    uint8_t device;
    uint8_t function;

    constexpr bool operator==(const pci_address& other) const {
        return this->domain == other.domain
            && this->bus == other.bus
            && this->device == other.device
            && this->function == other.function;
    }

    constexpr bool operator!=(const pci_address& other) const {
        return !(*this == other);
    }
};

template<>
struct std::formatter<pci_address, char> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FmtContext>
    FmtContext::iterator format(pci_address addr, FmtContext& ctx) const {
        return std::format_to(
            ctx.out(),
            "{:04x}:{:02x}:{:02x}.{:01x}",
            addr.domain,
            addr.bus,
            addr.device,
            addr.function
        );
    }
};

template <typename T>
struct stddev_helper {
    static T compute(const std::vector<T>& items, const T& average) {
        auto variance = T{0};
        for (const auto& item : items) {
            const auto diff = item - average;
            variance += diff * diff;
        }

        return std::sqrt(variance);
    }
};

template <typename Rep, typename Period>
struct stddev_helper<std::chrono::duration<Rep, Period>> {
    using Item = std::chrono::duration<Rep, Period>;

    static Item compute(const std::vector<Item>& items, const Item& average) {
        auto variance = Rep{0};
        for (const auto& item : items) {
            const auto diff = item.count() - average.count();
            variance += diff * diff;
        }

        return Item(std::sqrt(variance));
    }
};

template<typename T>
struct statistic {
    T average;
    T stddev;
    T largest;
    T smallest;

    explicit statistic(const std::vector<T>& items) {
        this->largest = items[0];
        this->smallest = items[0];
        auto total = items[0];

        for (size_t i = 1; i < items.size(); ++i) {
            const auto& item = items[i];
            this->largest = std::max(this->largest, item);
            this->smallest = std::min(this->smallest, item);
            total += item;
        }

        this->average = total / items.size();
        this->stddev = stddev_helper<T>::compute(items, this->average);
    }
};

template <typename T>
struct std::formatter<statistic<T>, char> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FmtContext>
    FmtContext::iterator format(const statistic<T>& stat, FmtContext& ctx) const {
        return std::format_to(
            ctx.out(),
            "{} +- {}σ [min {}, max {}]",
            stat.average,
            stat.stddev,
            stat.smallest,
            stat.largest
        );
    }
};

#endif
