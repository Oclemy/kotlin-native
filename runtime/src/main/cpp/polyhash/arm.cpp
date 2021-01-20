/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#include "polyhash/common.h"

#if defined(__arm__) or defined(__aarch64__)

alignas(32) constexpr auto p32 = DecreasingPowers<32>(31);   // [base^31, base^30, .., base^2, base, 1]
alignas(32) constexpr auto b32 = RepeatingPowers<8>(31, 32); // [base^32, base^32, .., base^32] (8)
alignas(32) constexpr auto b16 = RepeatingPowers<8>(31, 16); // [base^16, base^16, .., base^16] (8)
alignas(32) constexpr auto b8  = RepeatingPowers<8>(31, 8);  // [base^8,  base^8,  .., base^8 ] (8)
alignas(32) constexpr auto b4  = RepeatingPowers<8>(31, 4);  // [base^4,  base^4,  .., base^4 ] (8)

#include <arm_neon.h>

inline uint32x4_t squash(uint32x4_t x, uint32x4_t y) {
    uint32x4_t sum = vaddq_u32(x, y); // [x0 + y1, x1 + y1, x2 + y2, x3 + y3]
    return vdupq_n_u32(vaddvq_u32(sum)); // [x0..3 + y0..3, same, same, same]
}

inline void polyHashNeonUnalignedTail(int n, uint16_t const* str, uint32x4_t& res) {
    if (n == 0) return;

    uint32x4_t x4_7 = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str));
    uint32x4_t z4_7 = vmulq_u32(x4_7, *reinterpret_cast<uint32x4_t const*>(&p32[28])); // [b^3, b^2, b, 1]

    res = vmulq_u32(res, *reinterpret_cast<uint32x4_t const*>(&b4[0]));
    uint32x4_t sum = vdupq_n_u32(vaddvq_u32(z4_7));
    res = vaddq_u32(res, sum);
}

inline void polyHashNeonUnalignedUnroll32(int& n, uint16_t const*& str, uint32x4_t& res) {
    if (n < 8) return;

    // res0..res7 will accumulate 32 intermediate sums.
    uint32x4_t res0 = vdupq_n_u32(0);
    uint32x4_t res1 = vdupq_n_u32(0);
    uint32x4_t res2 = vdupq_n_u32(0);
    uint32x4_t res3 = vdupq_n_u32(0);
    uint32x4_t res4 = vdupq_n_u32(0);
    uint32x4_t res5 = vdupq_n_u32(0);
    uint32x4_t res6 = vdupq_n_u32(0);
    uint32x4_t res7 = vdupq_n_u32(0);

    do {
        uint32x4_t x0_3   = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str));
        uint32x4_t x4_7   = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 4));
        uint32x4_t x8_11  = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 8));
        uint32x4_t x12_15 = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 12));
        uint32x4_t x16_19 = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 16));
        uint32x4_t x20_23 = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 20));
        uint32x4_t x24_27 = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 24));
        uint32x4_t x28_31 = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 28));
        res0 = vmulq_u32(res0, *reinterpret_cast<uint32x4_t const*>(&b32[0]));
        res1 = vmulq_u32(res1, *reinterpret_cast<uint32x4_t const*>(&b32[0]));
        res2 = vmulq_u32(res2, *reinterpret_cast<uint32x4_t const*>(&b32[0]));
        res3 = vmulq_u32(res3, *reinterpret_cast<uint32x4_t const*>(&b32[0]));
        res4 = vmulq_u32(res4, *reinterpret_cast<uint32x4_t const*>(&b32[0]));
        res5 = vmulq_u32(res5, *reinterpret_cast<uint32x4_t const*>(&b32[0]));
        res6 = vmulq_u32(res6, *reinterpret_cast<uint32x4_t const*>(&b32[0]));
        res7 = vmulq_u32(res7, *reinterpret_cast<uint32x4_t const*>(&b32[0]));
        uint32x4_t z0_3   = vmulq_u32(x0_3,   *reinterpret_cast<uint32x4_t const*>(&p32[0]));  // [b^31, b^30, b^29, b^28]
        uint32x4_t z4_7   = vmulq_u32(x4_7,   *reinterpret_cast<uint32x4_t const*>(&p32[4]));  // [b^27, b^26, b^25, b^24]
        uint32x4_t z8_11  = vmulq_u32(x8_11,  *reinterpret_cast<uint32x4_t const*>(&p32[8]));  // [b^23, b^22, b^21, b^20]
        uint32x4_t z12_15 = vmulq_u32(x12_15, *reinterpret_cast<uint32x4_t const*>(&p32[12])); // [b^19, b^18, b^17, b^16]
        uint32x4_t z16_19 = vmulq_u32(x16_19, *reinterpret_cast<uint32x4_t const*>(&p32[16])); // [b^15, b^14, b^13, b^12]
        uint32x4_t z20_23 = vmulq_u32(x20_23, *reinterpret_cast<uint32x4_t const*>(&p32[20])); // [b^11, b^10, b^9,  b^8 ]
        uint32x4_t z24_27 = vmulq_u32(x24_27, *reinterpret_cast<uint32x4_t const*>(&p32[24])); // [b^7,  b^6,  b^5,  b^4 ]
        uint32x4_t z28_31 = vmulq_u32(x28_31, *reinterpret_cast<uint32x4_t const*>(&p32[28])); // [b^3,  b^2,  b,    1   ]
        res0 = vaddq_u32(res0, z0_3);
        res1 = vaddq_u32(res1, z4_7);
        res2 = vaddq_u32(res2, z8_11);
        res3 = vaddq_u32(res3, z12_15);
        res4 = vaddq_u32(res4, z16_19);
        res5 = vaddq_u32(res5, z20_23);
        res6 = vaddq_u32(res6, z24_27);
        res7 = vaddq_u32(res7, z28_31);

        str += 32;
        n -= 8;
    } while (n >= 8);

    res = vaddq_u32(res, vaddq_u32(vaddq_u32(squash(res0, res1), squash(res2, res3)),
                                   vaddq_u32(squash(res4, res5), squash(res6, res7))));
}

inline void polyHashNeonUnalignedUnroll16(int& n, uint16_t const*& str, uint32x4_t& res) {
    if (n < 4) return;

    res = vmulq_u32(res, *reinterpret_cast<uint32x4_t const*>(&b16[0]));

    // res0, res1, res2, res3 will accumulate 16 intermediate sums.
    uint32x4_t res0 = vdupq_n_u32(0);
    uint32x4_t res1 = vdupq_n_u32(0);
    uint32x4_t res2 = vdupq_n_u32(0);
    uint32x4_t res3 = vdupq_n_u32(0);

    do {
        uint32x4_t x0_3   = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str));
        uint32x4_t x4_7   = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 4));
        uint32x4_t x8_11  = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 8));
        uint32x4_t x12_15 = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 12));
        res0 = vmulq_u32(res0, *reinterpret_cast<uint32x4_t const*>(&b16[0]));
        res1 = vmulq_u32(res1, *reinterpret_cast<uint32x4_t const*>(&b16[0]));
        res2 = vmulq_u32(res2, *reinterpret_cast<uint32x4_t const*>(&b16[0]));
        res3 = vmulq_u32(res3, *reinterpret_cast<uint32x4_t const*>(&b16[0]));
        uint32x4_t z0_3   = vmulq_u32(x0_3,   *reinterpret_cast<uint32x4_t const*>(&p32[16])); // [b^15, b^14, b^13, b^12]
        uint32x4_t z4_7   = vmulq_u32(x4_7,   *reinterpret_cast<uint32x4_t const*>(&p32[20])); // [b^11, b^10, b^9,  b^8 ]
        uint32x4_t z8_11  = vmulq_u32(x8_11,  *reinterpret_cast<uint32x4_t const*>(&p32[24])); // [b^7,  b^6,  b^5,  b^4 ]
        uint32x4_t z12_15 = vmulq_u32(x12_15, *reinterpret_cast<uint32x4_t const*>(&p32[28])); // [b^3,  b^2,  b,    1   ]
        res0 = vaddq_u32(res0, z0_3);
        res1 = vaddq_u32(res1, z4_7);
        res2 = vaddq_u32(res2, z8_11);
        res3 = vaddq_u32(res3, z12_15);

        str += 16;
        n -= 4;
    } while (n >= 4);

    res = vaddq_u32(res, vaddq_u32(squash(res0, res1), squash(res2, res3)));
}

inline void polyHashNeonUnalignedUnroll8(int& n, uint16_t const*& str, uint32x4_t& res) {
    if (n < 2) return;

    res = vmulq_u32(res, *reinterpret_cast<uint32x4_t const*>(&b8[0]));

    // res0, res1 will accumulate 8 intermediate sums.
    uint32x4_t res0 = vdupq_n_u32(0);
    uint32x4_t res1 = vdupq_n_u32(0);

    do {
        uint32x4_t x0_3 = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str));
        uint32x4_t x4_7 = vmovl_u16(*reinterpret_cast<uint16x4_t const*>(str + 4));
        res0 = vmulq_u32(res0, *reinterpret_cast<uint32x4_t const*>(&b8[0]));
        res1 = vmulq_u32(res1, *reinterpret_cast<uint32x4_t const*>(&b8[0]));
        uint32x4_t z0_3 = vmulq_u32(x0_3, *reinterpret_cast<uint32x4_t const*>(&p32[24])); // [b^7, b^6, b^5, b^4]
        uint32x4_t z4_7 = vmulq_u32(x4_7, *reinterpret_cast<uint32x4_t const*>(&p32[28])); // [b^3, b^2, b,   1  ]
        res0 = vaddq_u32(res0, z0_3);
        res1 = vaddq_u32(res1, z4_7);

        str += 8;
        n -= 2;
    } while (n >= 2);

    res = vaddq_u32(res, squash(res0, res1));
}

int polyHashNeonUnalignedUnrollUpTo16(int n, uint16_t const* str) {
    uint32x4_t res = vdupq_n_u32(0);

    polyHashNeonUnalignedUnroll16(n, str, res);
    polyHashNeonUnalignedUnroll8(n, str, res);
    polyHashNeonUnalignedTail(n, str, res);

    return vgetq_lane_u32(res, 0);
}

int polyHashNeonUnalignedUnrollUpTo32(int n, uint16_t const* str) {
    uint32x4_t res = vdupq_n_u32(0);

    polyHashNeonUnalignedUnroll32(n, str, res);
    polyHashNeonUnalignedUnroll16(n, str, res);
    polyHashNeonUnalignedUnroll8(n, str, res);
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

int polyHash_arm(int length, uint16_t const* str) {
    if (!neonSupported) {
        // Vectorization is not supported.
        int res = 0;
        for (int i = 0; i < length; ++i)
            res = res * 31 + str[i];
        return res;
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

#endif