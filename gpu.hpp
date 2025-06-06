#ifndef _GPU_HPP
#define _GPU_HPP

#include "common.hpp"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <format>
#include <cstddef>
#include <cassert>

#define GPU_TRY(expr) {              \
    const auto _result = (expr);     \
    if (_result != hipSuccess) {     \
        throw ::gpu::error(_result); \
    }                                \
}

#define HSA_TRY(expr) {                  \
    const auto _result = (expr);         \
    if (_result != HSA_STATUS_SUCCESS) { \
        throw ::gpu::error(_result);     \
    }                                    \
}

#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx9_4_generic__)
    #define GPU_FAMILY_CDNA3
#elif defined(__gfx90a__)
    #define GPU_FAMILY_CDNA2
#elif defined(__gfx908__)
    #define GPU_FAMILY_CDNA1
#elif defined(__gfx900__) || defined(__gfx902__) || defined(__gfx904__) || defined(__gfx906__) \
    || defined(__gfx90c__) || defined(__gfx9_generic__)
    #define GPU_FAMILY_GCN5
#elif defined(__GFX12__) || defined(__gfx12_generic__)
    #define GPU_FAMILY_RDNA4
#elif defined(__GFX11__) || defined(__gfx11_generic__)
    #define GPU_FAMILY_RDNA3
#elif defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) \
    || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)                        \
    || defined(__gfx10_3_generic__)
    #define GPU_FAMILY_RDNA2
#elif defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1013__) \
    || defined(__gfx10_1_generic__)
    #define GPU_FAMILY_RDNA1
#elif defined(__SPIRV__)
    #define GPU_FAMILY_SPIRV
#elif defined(__HIP_DEVICE_COMPILE__)
    // Double check the build target for typos otherwise please submit an issue or pull request!
    #error "unknown build target"
#endif


namespace gpu {
    struct error: traced_error {
        hipError_t status;

        static const char* get_hsa_strerror(hsa_status_t status) {
            const char* msg;
            assert(hsa_status_string(status, &msg) == HSA_STATUS_SUCCESS);
            return msg;
        }

        error(hipError_t status):
            traced_error("{} ({})", hipGetErrorString(status), static_cast<int>(status)),
            status(status)
        {
            assert(status != hipSuccess);
        }

        error(hsa_status_t status):
            traced_error("{} ({})", get_hsa_strerror(status), static_cast<int>(status))
        {
            assert(status != HSA_STATUS_SUCCESS);
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

    enum class cache_level {
        l1,
        l2,
        l3,
        l4
    };

    struct device {
        // This structure contains a fixed-up version of the device's properties,
        // derived from both the HIP and HSA properties.
        struct properties {
            std::string device_name;
            std::string arch_name;
            pci_address pci_address;
            uint64_t total_global_mem;
            uint32_t warp_size;
            uint32_t compute_units;
            uint32_t simds_per_cu;
            uint32_t simd_width;
            uint32_t cacheline_size;
            uint32_t clock_rate;
            uint32_t cache_size[4];

            uint32_t total_simds() const {
                return this->compute_units * this->simds_per_cu;
            }

            uint32_t get_cache_size(cache_level level) const {
                return this->cache_size[static_cast<std::underlying_type_t<cache_level>>(level)];
            }

            uint32_t largest_cache_size() const {
                for (int i = std::size(this->cache_size) - 1; i >= 0; --i) {
                    if (this->cache_size[i] != 0) {
                        return this->cache_size[i];
                    }
                }
                // If we couldn't find the exact size of the largest cache possible,
                // 256 MB on the MI300.
                return 256 * 1024 * 1024;
            }
        };

        int hip_ordinal;
        hsa_agent_t hsa_agent;
        // Fetching the properties is relatively slow, so just cache them in this structure.
        properties properties;

        explicit device(int hip_ordinal):
            hip_ordinal(hip_ordinal)
        {
            hipDeviceProp_t hip_props;
            GPU_TRY(hipGetDeviceProperties(&hip_props, this->hip_ordinal));

            this->properties.device_name = hip_props.name;
            this->properties.arch_name = hip_props.gcnArchName;
            this->properties.total_global_mem = hip_props.totalGlobalMem;
            this->properties.warp_size = hip_props.warpSize;
            this->properties.clock_rate = hip_props.clockRate;

            this->properties.pci_address = {
                .domain = static_cast<uint16_t>(hip_props.pciDomainID),
                .bus = static_cast<uint8_t>(hip_props.pciBusID),
                .device = static_cast<uint8_t>(hip_props.pciDeviceID),
                .function = static_cast<uint8_t>(0), // There is no pciFunctionID, so presumably its always 0?
            };

            const auto iterate = [](hsa_agent_t agent, void* data) {
                auto& dev = *reinterpret_cast<device*>(data);
                hsa_status_t status;
                uint32_t pci_domain_id;
                status = hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN), &pci_domain_id);
                if (status != HSA_STATUS_SUCCESS) return status;

                uint32_t pci_bfd_id;
                status = hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &pci_bfd_id);
                if (status != HSA_STATUS_SUCCESS) return status;

                const auto hsa_addr = pci_address {
                    .domain = static_cast<uint16_t>(pci_domain_id & 0xFFFF),
                    .bus = static_cast<uint8_t>((pci_bfd_id >> 8) & ((1 << 8) - 1)),
                    .device = static_cast<uint8_t>((pci_bfd_id >> 3) & ((1 << 5) - 1)),
                    .function = static_cast<uint8_t>(pci_bfd_id & ((1 << 3) - 1)),
                };

                if (hsa_addr == dev.properties.pci_address) {
                    dev.hsa_agent = agent;
                    return HSA_STATUS_INFO_BREAK;
                }

                return HSA_STATUS_SUCCESS;
            };

            const auto status = hsa_iterate_agents(iterate, reinterpret_cast<void*>(this));
            if (status != HSA_STATUS_INFO_BREAK) {
                throw traced_error("could not map HIP device id {} to a HSA device", this->hip_ordinal);
            }

            const auto get_hsa_info = [&](const auto field, auto& value) {
                HSA_TRY(hsa_agent_get_info(this->hsa_agent, static_cast<hsa_agent_info_t>(field), &value));
            };

            get_hsa_info(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, this->properties.compute_units);
            get_hsa_info(HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU, this->properties.simds_per_cu);
            get_hsa_info(HSA_AMD_AGENT_INFO_CACHELINE_SIZE, this->properties.cacheline_size);
            get_hsa_info(HSA_AGENT_INFO_CACHE_SIZE, this->properties.cache_size);
        }

        void make_active() const {
            GPU_TRY(hipSetDevice(this->hip_ordinal));
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

        family_set get_family() const {
            const auto arch_name = this->properties.arch_name;
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

    __device__
    constexpr family_set get_device_family() {
        // See https://llvm.org/docs/AMDGPUUsage.html#instructions
        #ifdef GPU_FAMILY_CDNA3
            return family_set::cdna3;
        #elifdef GPU_FAMILY_CDNA2
            return family_set::cdna2;
        #elifdef GPU_FAMILY_CDNA1
            return family_set::cdna1;
        #elifdef GPU_FAMILY_GCN5
            return family_set::gcn5;
        #elifdef GPU_FAMILY_RDNA4
            return family_set::rdna4;
        #elifdef GPU_FAMILY_RDNA3
            return family_set::rdna3;
        #elifdef GPU_FAMILY_RDNA2
            return family_set::rdna2;
        #elifdef GPU_FAMILY_RDNA1
            return family_set::rdna1;
        #elifdef GPU_FAMILY_SPIRV
            return family_set::none; // For now
        #else
            // Make the compiler happy (this path is not reachable)
            return family_set::none;
        #endif
    }
}

#endif
