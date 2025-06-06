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

#endif
