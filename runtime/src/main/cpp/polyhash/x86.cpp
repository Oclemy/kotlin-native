/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#include "polyhash/common.h"
#include "polyhash/x86.h"

#if defined(__x86_64__) or defined(__i386__)

#include <immintrin.h>

#pragma clang attribute push (__attribute__((target("avx2"))), apply_to=function)

namespace {

alignas(32) constexpr auto p64 = DecreasingPowers<64>(31);   // [base^63, base^62, .., base^2, base, 1]
alignas(32) constexpr auto b64 = RepeatingPowers<8>(31, 64); // [base^64, base^64, .., base^64] (8)
alignas(32) constexpr auto b32 = RepeatingPowers<8>(31, 32); // [base^32, base^32, .., base^32] (8)
alignas(32) constexpr auto b16 = RepeatingPowers<8>(31, 16); // [base^16, base^16, .., base^16] (8)
alignas(32) constexpr auto b8  = RepeatingPowers<8>(31, 8);  // [base^8,  base^8,  .., base^8 ] (8)
alignas(32) constexpr auto b4  = RepeatingPowers<8>(31, 4);  // [base^4,  base^4,  .., base^4 ] (8)

inline __m128i squash(__m256i x, __m256i y) {
    __m256i sum = _mm256_hadd_epi32(x, y);         // [x0 + x1, x2 + x3, y0 + y1, y2 + y3, x4 + x5, x6 + x7, y4 + y5, y6 + y7]
    sum = _mm256_hadd_epi32(sum, sum);             // [x0..3, y0..3, x0..3, y0..3, x4..7, y4..7, x4..7, y4..7]
    sum = _mm256_hadd_epi32(sum, sum);             // [x0..3 + y0..3, same, same, same, x4..7 + y4..7, same, same, same]
    __m128i lo = _mm256_extracti128_si256(sum, 0); // [x0..3 + y0..3, same, same, same]
    __m128i hi = _mm256_extracti128_si256(sum, 1); // [x4..7 + y4..7, same, same, same]
    return _mm_add_epi32(lo, hi);                  // [x0..7 + y0..7, same, same, same]
}

inline __m128i squash(__m128i x, __m128i y) {
    __m128i sum = _mm_hadd_epi32(x, y); // [x0 + x1, x2 + x3, y0 + y1, y2 + y3]
    sum = _mm_hadd_epi32(sum, sum);     // [x0..3, y0..3, x0..3, y0..3]
    return _mm_hadd_epi32(sum, sum);    // [x0..3 + y0..3, same, same, same]
}

inline void polyHashSSEUnalignedTail(int n, uint16_t const* str, __m128i& res) {
    if (n == 0) return;

    __m128i x4_7 = _mm_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str));
    __m128i z4_7 = _mm_mullo_epi32(x4_7, *reinterpret_cast<__m128i const*>(&p64[60])); // [b^3, b^2, b, 1]

    res = _mm_mullo_epi32(res, *reinterpret_cast<__m128i const*>(&b4[0]));
    __m128i sum = _mm_hadd_epi32(z4_7, z4_7);
    sum = _mm_hadd_epi32(sum, sum);
    res = _mm_add_epi32(res, sum);
}

inline void polyHashAVX2UnalignedTail(int n, uint16_t const* str, __m128i& res) {
    if (n >= 2) {
        __m256i x0_7 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<__m128i const*>(str)));
        __m256i z0_7 = _mm256_mullo_epi32(x0_7, *reinterpret_cast<__m256i const*>(&p64[56])); // [b^7, .., b, 1]
        res = _mm_mullo_epi32(res, *reinterpret_cast<__m128i const*>(&b8[0]));
        __m256i sum = _mm256_hadd_epi32(z0_7, z0_7);
        sum = _mm256_hadd_epi32(sum, sum);
        res = _mm_add_epi32(res, _mm256_extracti128_si256(sum, 0));
        res = _mm_add_epi32(res, _mm256_extracti128_si256(sum, 1));

        str += 8;
        n -= 2;
    }

    polyHashSSEUnalignedTail(n, str, res);
}

int polyHashAVX2UnalignedUnrollUpTo16(int n, uint16_t const* str) {
    __m128i res = _mm_setzero_si128();

    polyHashUnroll2(n, str, res, __m256i, __m128i, __m128i, b16, p64, 48, _mm256_setzero_si256(), _mm256_cvtepu16_epi32,
                    _mm256_mullo_epi32, _mm256_add_epi32, _mm_mullo_epi32, _mm_add_epi32, squash);
    polyHashAVX2UnalignedTail(n, str, res);

    return _mm_cvtsi128_si32(res);
}

int polyHashAVX2UnalignedUnrollUpTo32(int n, uint16_t const* str) {
    __m128i res = _mm_setzero_si128();

    polyHashUnroll4(n, str, res, __m256i, __m128i, __m128i, b32, p64, 32, _mm256_setzero_si256(), _mm256_cvtepu16_epi32,
                    _mm256_mullo_epi32, _mm256_add_epi32, _mm_mullo_epi32, _mm_add_epi32, squash);
    polyHashUnroll2(n, str, res, __m256i, __m128i, __m128i, b16, p64, 48, _mm256_setzero_si256(), _mm256_cvtepu16_epi32,
                    _mm256_mullo_epi32, _mm256_add_epi32, _mm_mullo_epi32, _mm_add_epi32, squash);
    polyHashAVX2UnalignedTail(n, str, res);

    return _mm_cvtsi128_si32(res);
}

int polyHashAVX2UnalignedUnrollUpTo64(int n, uint16_t const* str) {
    __m128i res = _mm_setzero_si128();

    polyHashUnroll8(n, str, res, __m256i, __m128i, b64, p64, 0, _mm256_setzero_si256(), _mm256_cvtepu16_epi32,
                    _mm256_mullo_epi32, _mm256_add_epi32, _mm_add_epi32, squash);
    polyHashUnroll4(n, str, res, __m256i, __m128i, __m128i, b32, p64, 32, _mm256_setzero_si256(), _mm256_cvtepu16_epi32,
                    _mm256_mullo_epi32, _mm256_add_epi32, _mm_mullo_epi32, _mm_add_epi32, squash);
    polyHashUnroll2(n, str, res, __m256i, __m128i, __m128i, b16, p64, 48, _mm256_setzero_si256(), _mm256_cvtepu16_epi32,
                    _mm256_mullo_epi32, _mm256_add_epi32, _mm_mullo_epi32, _mm_add_epi32, squash);
    polyHashAVX2UnalignedTail(n, str, res);

    return _mm_cvtsi128_si32(res);
}

int polyHashSSEUnalignedUnrollUpTo8(int n, uint16_t const* str) {
    __m128i res = _mm_setzero_si128();

    polyHashUnroll2(n, str, res, __m128i, __m128i, __m128i, b8, p64, 56, _mm_setzero_si128(), _mm_cvtepu16_epi32,
                    _mm_mullo_epi32, _mm_add_epi32, _mm_mullo_epi32, _mm_add_epi32, squash);
    polyHashSSEUnalignedTail(n, str, res);

    return _mm_cvtsi128_si32(res);
}

int polyHashSSEUnalignedUnrollUpTo16(int n, uint16_t const* str) {
    __m128i res = _mm_setzero_si128();

    polyHashUnroll4(n, str, res, __m128i, __m128i, __m128i, b16, p64, 48, _mm_setzero_si128(), _mm_cvtepu16_epi32,
                    _mm_mullo_epi32, _mm_add_epi32, _mm_mullo_epi32, _mm_add_epi32, squash);
    polyHashUnroll2(n, str, res, __m128i, __m128i, __m128i, b8, p64, 56, _mm_setzero_si128(), _mm_cvtepu16_epi32,
                    _mm_mullo_epi32, _mm_add_epi32, _mm_mullo_epi32, _mm_add_epi32, squash);
    polyHashSSEUnalignedTail(n, str, res);

    return _mm_cvtsi128_si32(res);
}

#if defined(__x86_64__)
    const bool x64 = true;
#else
    const bool x64 = false;
#endif
    bool initialized = false;
    bool sseSupported;
    bool avx2Supported;

}

int polyHash_x86(int length, uint16_t const* str) {
    if (!initialized) {
        initialized = true;
        sseSupported = __builtin_cpu_supports("sse4.1");
        avx2Supported = __builtin_cpu_supports("avx2");
    }
    if (length < 16 || (!sseSupported && !avx2Supported)) {
        // Either vectorization is not supported or the string is too short to gain from it.
        return polyHash_naive(length, str);
    }
    int res;
    if (length < 32)
        res = polyHashSSEUnalignedUnrollUpTo8(length / 4, str);
    else if (!avx2Supported)
        res = polyHashSSEUnalignedUnrollUpTo16(length / 4, str);
    else if (length < 128)
        res = polyHashAVX2UnalignedUnrollUpTo16(length / 4, str);
    else if (!x64 || length < 576)
        res = polyHashAVX2UnalignedUnrollUpTo32(length / 4, str);
    else // Such big unrolling requires 64-bit mode (in 32-bit mode there are only 8 vector registers)
        res = polyHashAVX2UnalignedUnrollUpTo64(length / 4, str);

    // Handle the tail naively.
    for (int i = length & 0xFFFFFFFC; i < length; ++i)
        res = res * 31 + str[i];
    return res;
}

#pragma clang attribute pop

#endif
