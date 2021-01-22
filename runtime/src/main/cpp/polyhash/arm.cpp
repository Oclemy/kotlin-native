/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#include "polyhash/common.h"
#include "polyhash/arm.h"

#if defined(__arm__) or defined(__aarch64__)

#ifndef __ARM_NEON

int polyHash_arm(int length, uint16_t const* str) {
    return polyHash_naive(length, str);
}

#else

#include <arm_neon.h>

namespace {

alignas(32) constexpr auto p32 = DecreasingPowers<32>(31);   // [base^31, base^30, .., base^2, base, 1]
alignas(32) constexpr auto b32 = RepeatingPowers<8>(31, 32); // [base^32, base^32, .., base^32] (8)
alignas(32) constexpr auto b16 = RepeatingPowers<8>(31, 16); // [base^16, base^16, .., base^16] (8)
alignas(32) constexpr auto b8  = RepeatingPowers<8>(31, 8);  // [base^8,  base^8,  .., base^8 ] (8)
alignas(32) constexpr auto b4  = RepeatingPowers<8>(31, 4);  // [base^4,  base^4,  .., base^4 ] (8)

inline uint32x4_t squash(uint32x4_t z) {
#ifdef __aarch64__
    return vdupq_n_u32(vaddvq_u32(z)); // [z0..3, same, same, same]
#else
    uint32x2_t lo = vget_low_u32(z);   // [z0, z1]
    uint32x2_t hi = vget_high_u32(z);  // [z2, z3]
    uint32x2_t sum = vadd_u32(lo, hi); // [z0 + z2, z1 + z3]
    sum = vpadd_u32(sum, sum);         // [z0..3, same]
    return vcombine_u32(sum, sum);     // [z0..3, same, same, same]
#endif
}

inline uint32x4_t squash(uint32x4_t x, uint32x4_t y) {
    return squash(vaddq_u32(x, y));
}

inline void polyHashNeonUnalignedTail(int n, uint16_t const* str, uint32x4_t& res) {
    if (n == 0) return;

    uint32x4_t x4_7 = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str));
    res = vmulq_u32(res, *reinterpret_cast<uint32x4_t const*>(&b4[0]));
    uint32x4_t z4_7 = vmulq_u32(x4_7, *reinterpret_cast<uint32x4_t const*>(&p32[28])); // [b^3, b^2, b, 1]
    res = vaddq_u32(res, squash(z4_7));
}

int polyHashNeonUnalignedUnrollUpTo16(int n, uint16_t const* str) {
    uint32x4_t res = vdupq_n_u32(0);

    polyHashUnroll4(n, str, res, uint32x4_t, uint32x4_t, uint16x4_t, b16, p32, 16, vdupq_n_u32(0),
                    vmovl_u16, vmulq_u32, vaddq_u32, vmulq_u32, vaddq_u32, squash);
    polyHashUnroll2(n, str, res, uint32x4_t, uint32x4_t, uint16x4_t, b8, p32, 24, vdupq_n_u32(0),
                    vmovl_u16, vmulq_u32, vaddq_u32, vmulq_u32, vaddq_u32, squash);
    polyHashNeonUnalignedTail(n, str, res);

    return vgetq_lane_u32(res, 0);
}

int polyHashNeonUnalignedUnrollUpTo32(int n, uint16_t const* str) {
    uint32x4_t res = vdupq_n_u32(0);

    polyHashUnroll8(n, str, res, uint32x4_t, uint16x4_t, b32, p32, 0, vdupq_n_u32(0),
                    vmovl_u16, vmulq_u32, vaddq_u32, vaddq_u32, squash);
    polyHashUnroll4(n, str, res, uint32x4_t, uint32x4_t, uint16x4_t, b16, p32, 16, vdupq_n_u32(0),
                    vmovl_u16, vmulq_u32, vaddq_u32, vmulq_u32, vaddq_u32, squash);
    polyHashUnroll2(n, str, res, uint32x4_t, uint32x4_t, uint16x4_t, b8, p32, 24, vdupq_n_u32(0),
                    vmovl_u16, vmulq_u32, vaddq_u32, vmulq_u32, vaddq_u32, squash);
    polyHashNeonUnalignedTail(n, str, res);

    return vgetq_lane_u32(res, 0);
}

#if defined(__aarch64__)
    const bool neonSupported = true; // AArch64 always supports Neon.
#elif defined(__ANDROID__)
    #include <cpu-features.h>
    const bool neonSupported = android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_NEON;
#elif defined(__APPLE__)
    const bool neonSupported = true; // It is supported starting from iPhone 3GS.
#elif defined(__linux__) or defined(__unix__)
    #include <sys/auxv.h>
    #include <asm/hwcap.h>
    const bool neonSupported = getauxval(AT_HWCAP) & HWCAP_NEON;
#else
    #error "Not supported"
#endif

}

int polyHash_arm(int length, uint16_t const* str) {
    if (!neonSupported) {
        // Vectorization is not supported.
        return polyHash_naive(length, str);
    }
    int res;
    if (length < 488)
        res = polyHashNeonUnalignedUnrollUpTo16(length / 4, str);
    else
        res = polyHashNeonUnalignedUnrollUpTo32(length / 4, str);
    // Handle the tail naively.
    for (int i = length & 0xFFFFFFFC; i < length; ++i)
        res = res * 31 + str[i];
    return res;
}

#endif // __ARM_NEON

#endif // defined(__arm__) or defined(__aarch64__)