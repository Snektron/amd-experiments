#ifndef _GPU_HPP
#define _GPU_HPP

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <sstream>
#include <stacktrace>
#include <cstddef>
#include <cassert>

#define GPU_TRY(expr) {                                               \
    const auto _result = (expr);                                      \
    if (_result != hipSuccess) {                                      \
        throw ::gpu::error(_result, std::stacktrace::current()); \
    }                                                                 \
}

namespace gpu {
    struct error: std::runtime_error {
        hipError_t status;
        std::stacktrace trace;

        static std::string get_error_msg(hipError_t status) {
            std::stringstream ss;
            ss << hipGetErrorString(status) << " (" << status << ")";
            return ss.str();
        }

        error(hipError_t status, std::stacktrace trace):
            runtime_error(get_error_msg(status)),
            status(status),
            trace(trace)
        {
            assert(status != hipSuccess);
        }
    };

    template <typename T>
    struct ptr {
        friend struct device;

        T* raw;

    private:
        ptr(size_t size) {
            GPU_TRY(hipMalloc(&this->raw, size * sizeof(T)));
        }

    public:

        ptr(const ptr&) = delete;
        ptr& operator=(const ptr&) = delete;

        ptr(ptr&& other):
            raw(std::exchange(other.raw, nullptr))
        {}

        ptr& operator=(ptr&& other) {
            std::swap(this->raw, other.raw);
            return *this;
        }

        ~ptr() {
            if (this->raw) {
                (void) hipFree(this->raw);
            }
        }
    };

    struct event {
        using duration = std::chrono::duration<float, std::milli>;

        hipEvent_t handle;

        explicit event() {
            GPU_TRY(hipEventCreate(&this->handle));
        }

        event(const event&) = delete;
        event& operator=(const event&) = delete;

        event(event&& other):
            handle(std::exchange(other.handle, nullptr))
        {}

        event& operator=(event&& other) {
            std::swap(this->handle, other.handle);
            return *this;
        }

        ~event() {
            if (this->handle) {
                (void) hipEventDestroy(this->handle);
            }
        }

        static duration elapsed(const event& start, const event& stop) {
            float ms;
            GPU_TRY(hipEventElapsedTime(&ms, start.handle, stop.handle));
            return duration(ms);
        }
    };

    struct launch_config {
        dim3 grid_size = dim3(1);
        dim3 block_size = dim3(1);
        unsigned int shared_mem_per_block = 0;
    };

    struct stream {
        friend struct device;

        enum class flags {
            none = 0,
            non_blocking = hipStreamNonBlocking,

            default_flags = hipStreamDefault,
        };

        hipStream_t handle;

    private:
        explicit stream(flags flags) {
            GPU_TRY(hipStreamCreateWithFlags(&this->handle, static_cast<unsigned int>(flags)));
        }

    public:
        stream(const stream&) = delete;
        stream& operator=(const stream&) = delete;

        stream(stream&& other):
            handle(std::exchange(other.handle, nullptr))
        {}

        stream& operator=(stream&& other) {
            std::swap(this->handle, other.handle);
            return *this;
        }

        ~stream() {
            if (this->handle) {
                (void) hipStreamDestroy(this->handle);
            }
        }

        void sync() const {
            GPU_TRY(hipStreamSynchronize(this->handle));
        }

        template <typename F, typename... Args>
        void launch(launch_config cfg, F f, Args&&... args) const {
            f<<<cfg.grid_size, cfg.block_size, cfg.shared_mem_per_block, this->handle>>>(std::forward<Args>(args)...);
            GPU_TRY(hipGetLastError());
        }

        void record(const event& event) const {
            GPU_TRY(hipEventRecord(event.handle, this->handle));
        }

        void memset(void* d_ptr, int ch, size_t count) const {
            GPU_TRY(hipMemsetAsync(d_ptr, ch, count));
        }
    };

    struct device {
        int ordinal;

        constexpr explicit device(int ordinal):
            ordinal(ordinal)
        { }

        void make_active() const {
            GPU_TRY(hipSetDevice(this->ordinal));
        }

        template <typename T>
        ptr<T> alloc(size_t size) const {
            this->make_active();
            return ptr<T>(size);
        }

        stream create_stream(stream::flags flags = stream::flags::default_flags) const {
            this->make_active();
            return stream(flags);
        }

        hipDeviceProp_t get_properties() const {
            hipDeviceProp_t props;
            GPU_TRY(hipGetDeviceProperties(&props, this->ordinal));
            return props;
        }

        size_t largest_cache_size() const {
            // Currently, HIP does not have a way to query the size
            // of infinity cache if it is present. For now, just return
            // the size of the maximum cache on any device, which is
            // 256 MiB on MI300X.
            return 256 * 1024 * 1024;
        }

        void sync() const {
            this->make_active();
            GPU_TRY(hipDeviceSynchronize());
        }
    };

    constexpr const static auto default_device = device(0);
}

#endif
