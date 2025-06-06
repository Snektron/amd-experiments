#ifndef _GPU_HPP
#define _GPU_HPP

#include "common.hpp"

#include <hip/hip_runtime.h>
#include <format>
#include <cstddef>
#include <cassert>

#define GPU_TRY(expr) {              \
    const auto _result = (expr);     \
    if (_result != hipSuccess) {     \
        throw ::gpu::error(_result); \
    }                                \
}

namespace gpu {
    struct error: traced_error {
        hipError_t status;

        error(hipError_t status):
            traced_error("{} ({})", hipGetErrorString(status), static_cast<int>(status)),
            status(status)
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
        explicit ptr(T* raw): raw(raw) {}

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

        constexpr explicit stream(hipStream_t underlying): handle(underlying) {}

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

    using device_properties = hipDeviceProp_t;

    struct family_set {
        enum bits {
            none = 0x0,

            gcn5 = 0x1,

            rdna1 = 0x2,
            rdna2 = 0x4,
            rdna3 = 0x8,
            rdna4 = 0x10,

            cdna1 = 0x20,
            cdna2 = 0x40,
            cdna3 = 0x80,

            all = (cdna3 << 1) - 1
        };

        using backing_type = std::underlying_type_t<bits>;

        bits families;

        constexpr family_set(bits families): families(families) {}

        constexpr bool operator==(const family_set& other) const {
            return this->families == other.families;
        }

        constexpr bool operator!=(const family_set& other) const {
            return !(*this == other);
        }

        template <typename F>
        __host__ __device__
        constexpr static family_set unop(family_set a, F f) {
            return static_cast<bits>(f(static_cast<backing_type>(a.families)));
        }

        constexpr family_set operator~() const {
            return unop(*this, [](auto a) { return ~a & bits::all; });
        }

        template <typename F>
        __host__ __device__
        constexpr static family_set binop(family_set a, family_set b, F f) {
            return static_cast<bits>(f(static_cast<backing_type>(a.families), static_cast<backing_type>(b.families)));
        }

        constexpr family_set operator|(family_set other) const {
            return binop(*this, other, [](auto a, auto b) { return a | b; });
        }

        constexpr family_set& operator|=(family_set other) {
            *this = binop(*this, other, [](auto a, auto b) { return a | b; });
            return *this;
        }

        constexpr family_set operator&(family_set other) const {
            return binop(*this, other, [](auto a, auto b) { return a & b; });
        }

        constexpr family_set& operator&(family_set other) {
            *this = binop(*this, other, [](auto a, auto b) { return a & b; });
            return *this;
        }

        constexpr family_set operator^(family_set other) const {
            return binop(*this, other, [](auto a, auto b) { return a ^ b; });
        }

        constexpr family_set& operator^(family_set other) {
            *this = binop(*this, other, [](auto a, auto b) { return a ^ b; });
            return *this;
        }

        constexpr bool contains(family_set other) const {
            return (*this & other) == other;
        }
    };

    struct device {
        int ordinal;
        // Fetching the properties is relatively slow, so just cache them in this structure.
        device_properties properties;

        explicit device(int ordinal):
            ordinal(ordinal)
        {
            GPU_TRY(hipGetDeviceProperties(&this->properties, this->ordinal));
        }

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

        size_t largest_cache_size() const {
            // Currently, HIP does not have a way to query the size
            // of infinity cache if it is present. For now, just return
            // the size of the maximum cache on any device, which is
            // 256 MiB on MI300X.
            return 256 * 1024 * 1024;
        }

        pci_address get_pci_bus_id() const {
            char pci_string[256] = {0};
            GPU_TRY(hipDeviceGetPCIBusId(pci_string, sizeof(pci_string) - 1, this->ordinal));
            pci_address addr;
            unsigned int dom, bus, dev, func;
            if (std::sscanf(pci_string, "%04x:%02x:%02x.%01x", &dom, &bus, &dev, &func) != 4) {
                throw traced_error("could not parse GPU {} PCI id '{}'", this->ordinal, pci_string);
            }
            return {
                .domain = static_cast<uint16_t>(dom),
                .bus = static_cast<uint8_t>(bus),
                .device = static_cast<uint8_t>(dev),
                .function = static_cast<uint8_t>(func),
            };
        }

        family_set get_family() const {
            const std::string_view arch_name = this->properties.gcnArchName;
            if (arch_name.starts_with("gfx12")) {
                return family_set::rdna4;
            } else if (arch_name.starts_with("gfx11")) {
                return family_set::rdna3;
            } else if (arch_name.starts_with("gfx103")) {
                return family_set::rdna2;
            } else if (arch_name.starts_with("gfx101")) {
                return family_set::rdna1;
            } else if (arch_name.starts_with("gfx94") || arch_name.starts_with("gfx95")) {
                return family_set::cdna3;
            } else if (arch_name.starts_with("gfx90a")) {
                return family_set::cdna2;
            } else if (arch_name.starts_with("gfx908")) {
                return family_set::cdna1;
            } else if (arch_name.starts_with("gfx9")) {
                return family_set::gcn5;
            }

            return family_set::none;
        }

        void sync() const {
            this->make_active();
            GPU_TRY(hipDeviceSynchronize());
        }
    };

    static device get_default_device() {
        return device(0);
    }
}

template<>
struct std::formatter<gpu::pci_address, char> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template <typename FmtContext>
    FmtContext::iterator format(gpu::pci_address addr, FmtContext& ctx) const {
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
