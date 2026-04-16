#pragma once

#include <CL/cl.h>

#include <cstdint>

struct ggml_opencl_extra_base_view {
    cl_mem data_device = nullptr;
    cl_ulong offset = 0;
};

constexpr uint32_t GGML_OPENCL_EXTRA_MAGIC_BASE  = 0x434c4230u; // CLB0
constexpr uint32_t GGML_OPENCL_EXTRA_MAGIC_Q4_0  = 0x51343030u; // Q400
constexpr uint32_t GGML_OPENCL_EXTRA_MAGIC_Q4_1  = 0x51343130u; // Q410
constexpr uint32_t GGML_OPENCL_EXTRA_MAGIC_MXFP4 = 0x4d584634u; // MXF4
constexpr uint32_t GGML_OPENCL_EXTRA_MAGIC_Q8_0  = 0x51383030u; // Q800
constexpr uint32_t GGML_OPENCL_EXTRA_MAGIC_Q6_K  = 0x51364b30u; // Q6K0

struct ggml_opencl_extra_header {
    uint32_t magic = 0;
    cl_mem base_data_device = nullptr;
    cl_ulong base_offset = 0;
    const void * owner_buffer = nullptr;
};

inline bool ggml_opencl_extra_header_has_known_magic(uint32_t magic) {
    switch (magic) {
        case GGML_OPENCL_EXTRA_MAGIC_BASE:
        case GGML_OPENCL_EXTRA_MAGIC_Q4_0:
        case GGML_OPENCL_EXTRA_MAGIC_Q4_1:
        case GGML_OPENCL_EXTRA_MAGIC_MXFP4:
        case GGML_OPENCL_EXTRA_MAGIC_Q8_0:
        case GGML_OPENCL_EXTRA_MAGIC_Q6_K:
            return true;
        default:
            return false;
    }
}

inline void ggml_opencl_extra_header_reset(ggml_opencl_extra_header & header, uint32_t magic) {
    header.magic = magic;
    header.base_data_device = nullptr;
    header.base_offset = 0;
    header.owner_buffer = nullptr;
}

inline void ggml_opencl_extra_header_bind_base(
        ggml_opencl_extra_header & header,
        const void * owner_buffer,
        cl_mem data_device,
        cl_ulong offset) {
    header.owner_buffer = owner_buffer;
    header.base_data_device = data_device;
    header.base_offset = offset;
}

inline bool ggml_opencl_extra_header_matches_owner(
        const ggml_opencl_extra_header & header,
        const void * owner_buffer) {
    return owner_buffer == nullptr || header.owner_buffer == owner_buffer;
}

inline bool ggml_opencl_extra_header_can_reuse(
        const ggml_opencl_extra_header & header,
        const void * owner_buffer = nullptr) {
    return ggml_opencl_extra_header_has_known_magic(header.magic) &&
           ggml_opencl_extra_header_matches_owner(header, owner_buffer) &&
           header.base_data_device != nullptr;
}

inline bool ggml_opencl_extra_header_can_reuse(
        const ggml_opencl_extra_header & header,
        const void * owner_buffer,
        cl_ulong expected_base_offset) {
    return ggml_opencl_extra_header_can_reuse(header, owner_buffer) &&
           header.base_offset == expected_base_offset;
}

inline bool ggml_opencl_extra_header_can_reuse_base_extra(
        const ggml_opencl_extra_header & header,
        const void * owner_buffer = nullptr) {
    return header.magic == GGML_OPENCL_EXTRA_MAGIC_BASE &&
           ggml_opencl_extra_header_can_reuse(header, owner_buffer);
}

inline ggml_opencl_extra_base_view ggml_opencl_extra_header_base_view(
        const ggml_opencl_extra_header & header,
        const void * owner_buffer = nullptr) {
    if (!ggml_opencl_extra_header_has_known_magic(header.magic) ||
        !ggml_opencl_extra_header_matches_owner(header, owner_buffer) ||
        header.base_data_device == nullptr) {
        return {};
    }

    return {
        /* .data_device = */ header.base_data_device,
        /* .offset      = */ header.base_offset,
    };
}

inline ggml_opencl_extra_base_view ggml_opencl_extra_header_view(
        const ggml_opencl_extra_header & header,
        const void * owner_buffer,
        cl_ulong view_offset) {
    ggml_opencl_extra_base_view view = ggml_opencl_extra_header_base_view(header, owner_buffer);
    if (view.data_device == nullptr) {
        return {};
    }

    view.offset += view_offset;
    return view;
}
