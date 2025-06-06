#ifndef _COMMON_H
#define _COMMON_H

#include <stacktrace>
#include <sstream>
#include <format>
#include <string>
#include <cstdint>

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
        return std::format_to( ctx.out(),
            "{:04x}:{:02x}:{:02x}.{:01x}",
            addr.domain,
            addr.bus,
            addr.device,
            addr.function
        );
    }
};

#endif
