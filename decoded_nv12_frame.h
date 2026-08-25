#ifndef DECODED_NV12_FRAME_H
#define DECODED_NV12_FRAME_H

#include "qcap.h"
#include "qcap2.h"
#include "qcap2.nvbuf.h"

#include <cstring>

struct DecodedNv12Info {
    ULONG width = 0;
    ULONG height = 0;
    ULONG length = 0;
};

inline bool copyNv12Planes(const uint8_t* yPlane,
                           int yStride,
                           const uint8_t* uvPlane,
                           int uvStride,
                           ULONG width,
                           ULONG height,
                           BYTE* destination,
                           ULONG destinationCapacity)
{
    if (!yPlane || !uvPlane || !destination || width == 0 || height == 0 ||
        (width & 1U) != 0 || (height & 1U) != 0 ||
        yStride < static_cast<int>(width) || uvStride < static_cast<int>(width)) {
        return false;
    }

    const ULONG required = width * height * 3U / 2U;
    if (required > destinationCapacity) {
        return false;
    }

    BYTE* dstY = destination;
    BYTE* dstUV = destination + width * height;
    for (ULONG row = 0; row < height; ++row) {
        std::memcpy(dstY + row * width, yPlane + row * yStride, width);
    }
    for (ULONG row = 0; row < height / 2U; ++row) {
        std::memcpy(dstUV + row * width, uvPlane + row * uvStride, width);
    }
    return true;
}

inline bool copyPlanar420ToNv12(const uint8_t* yPlane,
                                int yStride,
                                const uint8_t* uPlane,
                                int uStride,
                                const uint8_t* vPlane,
                                int vStride,
                                ULONG width,
                                ULONG height,
                                BYTE* destination,
                                ULONG destinationCapacity)
{
    if (!yPlane || !uPlane || !vPlane || !destination || width == 0 || height == 0 ||
        (width & 1U) != 0 || (height & 1U) != 0 ||
        yStride < static_cast<int>(width) ||
        uStride < static_cast<int>(width / 2U) ||
        vStride < static_cast<int>(width / 2U)) {
        return false;
    }

    const ULONG required = width * height * 3U / 2U;
    if (required > destinationCapacity) {
        return false;
    }

    for (ULONG row = 0; row < height; ++row) {
        std::memcpy(destination + row * width, yPlane + row * yStride, width);
    }

    BYTE* dstUV = destination + width * height;
    for (ULONG row = 0; row < height / 2U; ++row) {
        const uint8_t* srcU = uPlane + row * uStride;
        const uint8_t* srcV = vPlane + row * vStride;
        BYTE* dst = dstUV + row * width;
        for (ULONG column = 0; column < width / 2U; ++column) {
            dst[column * 2U] = srcU[column];
            dst[column * 2U + 1U] = srcV[column];
        }
    }
    return true;
}

inline bool isEightBitNv12(NvBufSurfaceColorFormat format)
{
    return format == NVBUF_COLOR_FORMAT_NV12 ||
           format == NVBUF_COLOR_FORMAT_NV12_ER ||
           format == NVBUF_COLOR_FORMAT_NV12_709 ||
           format == NVBUF_COLOR_FORMAT_NV12_709_ER ||
           format == NVBUF_COLOR_FORMAT_NV12_2020;
}

inline bool copyMappedNv12Surface(qcap2_rcbuffer_t* frameBuffer,
                                   NvBufSurface* surface,
                                   BYTE* destination,
                                   ULONG destinationCapacity,
                                   DecodedNv12Info* info)
{
    if (!surface || !surface->surfaceList || surface->numFilled == 0) {
        return false;
    }

    NvBufSurfaceParams& params = surface->surfaceList[0];
    if (!isEightBitNv12(params.colorFormat) || params.width == 0 || params.height == 0 ||
        params.planeParams.num_planes < 2) {
        return false;
    }

    if (qcap2_rcbuffer_map_nvbuf(frameBuffer, NVBUF_MAP_READ) != QCAP_RS_SUCCESSFUL) {
        return false;
    }

    bool copied = false;
    if (qcap2_rcbuffer_sync_nvbuf_for_cpu(frameBuffer) == QCAP_RS_SUCCESSFUL) {
        const uint8_t* yPlane = static_cast<const uint8_t*>(params.mappedAddr.addr[0]);
        const uint8_t* uvPlane = static_cast<const uint8_t*>(params.mappedAddr.addr[1]);
        if (!uvPlane && yPlane) {
            uvPlane = yPlane + params.planeParams.offset[1];
        }
        copied = copyNv12Planes(yPlane, params.planeParams.pitch[0],
                                uvPlane, params.planeParams.pitch[1],
                                params.width, params.height,
                                destination, destinationCapacity);
    }
    qcap2_rcbuffer_unmap_nvbuf(frameBuffer);

    if (copied) {
        info->width = params.width;
        info->height = params.height;
        info->length = params.width * params.height * 3U / 2U;
    }
    return copied;
}

// Copies a decoder-owned NV12 frame into a tightly packed CPU buffer without
// resizing it. Software-decoder frames are read through qcap2_av_frame_t;
// Jetson hardware-decoder frames are mapped directly from NvBufSurface.
inline bool copyDecodedNv12Frame(qcap2_rcbuffer_t* frameBuffer,
                                 ULONG fallbackWidth,
                                 ULONG fallbackHeight,
                                 BYTE* destination,
                                 ULONG destinationCapacity,
                                 DecodedNv12Info* info)
{
    if (!frameBuffer || !destination || !info) {
        return false;
    }

    NvBufSurface* nativeSurface = nullptr;
    if (qcap2_rcbuffer_get_nvbuf(frameBuffer, &nativeSurface) == QCAP_RS_SUCCESSFUL && nativeSurface) {
        return copyMappedNv12Surface(frameBuffer, nativeSurface, destination, destinationCapacity, info);
    }

    ULONG colorSpace = QCAP_COLORSPACE_TYPE_NV12;
    ULONG width = fallbackWidth;
    ULONG height = fallbackHeight;
    bool copied = false;

    PVOID lockedData = qcap2_rcbuffer_lock_data(frameBuffer);
    if (lockedData) {
        qcap2_av_frame_t* avFrame = reinterpret_cast<qcap2_av_frame_t*>(lockedData);
        uint8_t* planes[4] = {nullptr, nullptr, nullptr, nullptr};
        int strides[4] = {0, 0, 0, 0};
        qcap2_av_frame_get_video_property(avFrame, &colorSpace, &width, &height);
        qcap2_av_frame_get_buffer1(avFrame, planes, strides);

        if (width == 0 || height == 0) {
            width = fallbackWidth;
            height = fallbackHeight;
        }

        uint8_t* uvPlane = planes[1];
        int uvStride = strides[1];
        if (!uvPlane && planes[0] && strides[0] > 0 && height > 0) {
            uvPlane = planes[0] + static_cast<size_t>(strides[0]) * height;
            uvStride = strides[0];
        }

        if (colorSpace == QCAP_COLORSPACE_TYPE_NV12) {
            copied = copyNv12Planes(planes[0], strides[0], uvPlane, uvStride,
                                    width, height, destination, destinationCapacity);
        } else if (colorSpace == QCAP_COLORSPACE_TYPE_I420 ||
                   colorSpace == QCAP_COLORSPACE_TYPE_YV12) {
            const bool isI420 = colorSpace == QCAP_COLORSPACE_TYPE_I420;
            const uint8_t* uPlane = planes[isI420 ? 1 : 2];
            const uint8_t* vPlane = planes[isI420 ? 2 : 1];
            const int uStride = strides[isI420 ? 1 : 2];
            const int vStride = strides[isI420 ? 2 : 1];
            copied = copyPlanar420ToNv12(planes[0], strides[0],
                                         uPlane, uStride, vPlane, vStride,
                                         width, height, destination, destinationCapacity);
        }
        qcap2_rcbuffer_unlock_data(frameBuffer);
    }

    if (!copied) {
        return false;
    }

    info->width = width;
    info->height = height;
    info->length = width * height * 3U / 2U;
    return true;
}

#endif // DECODED_NV12_FRAME_H
