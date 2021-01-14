/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#include "polyhash/common.h"

#if defined(__x86_64__) or defined(__i386__)

namespace {

DecreasingPowers<31, 64> p64;   // [base^63, base^62, .., base^2, base, 1]
RepeatingPowers<31, 8, 64> b64; // [base^64, base^64, .., base^64] (8)
RepeatingPowers<31, 8, 32> b32; // [base^32, base^32, .., base^32] (8)
RepeatingPowers<31, 8, 16> b16; // [base^16, base^16, .., base^16] (8)
RepeatingPowers<31, 8, 8>  b8;  // [base^8 , base^8 , .., base^8]  (8)
RepeatingPowers<31, 8, 4>  b4;  // [base^4 , base^4 , .., base^4]  (8)

#include <immintrin.h>

inline __m256i squash(__m256i x, __m256i y) {
  __m256i sum = _mm256_hadd_epi32(x, y); // [x0 + x1, x2 + x3, y0 + y1, y2 + y3, x4 + x5, x6 + x7, y4 + y5, y6 + y7]
  sum = _mm256_hadd_epi32(sum, sum);     // [x0..3, y0..3, x0..3, y0..3, x4..7, y4..7, x4..7, y4..7]
  return _mm256_hadd_epi32(sum, sum);    // [x0..3 + y0..3, same, same, same, x4..7 + y4..7, same, same, same]
}

inline __m128i squash(__m128i x, __m128i y) {
  __m128i sum = _mm_hadd_epi32(x, y); // [x0 + x1, x2 + x3, y0 + y1, y2 + y3]
  sum = _mm_hadd_epi32(sum, sum);     // [x0..3, y0..3, x0..3, y0..3]
  return _mm_hadd_epi32(sum, sum);    // [x0..3 + y0..3, same, same, same]
}

inline void polyHashAVX2UnalignedUnroll64(int& n, uint16_t const*& str, __m128i& res) {
  if (n < 16) return;

  // res0..res7 will accumulate 64 intermediate sums.
  __m256i res0 = _mm256_setzero_si256();
  __m256i res1 = _mm256_setzero_si256();
  __m256i res2 = _mm256_setzero_si256();
  __m256i res3 = _mm256_setzero_si256();
  __m256i res4 = _mm256_setzero_si256();
  __m256i res5 = _mm256_setzero_si256();
  __m256i res6 = _mm256_setzero_si256();
  __m256i res7 = _mm256_setzero_si256();

  do {
    __m256i x0_7   = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str));
    __m256i x8_15  = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 8));
    __m256i x16_23 = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 16));
    __m256i x24_31 = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 24));
    __m256i x32_39 = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 32));
    __m256i x40_47 = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 40));
    __m256i x48_55 = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 48));
    __m256i x56_63 = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 56));
    res0 = _mm256_mullo_epi32(res0, *reinterpret_cast<__m256i*>(b64.values));
    res1 = _mm256_mullo_epi32(res1, *reinterpret_cast<__m256i*>(b64.values));
    res2 = _mm256_mullo_epi32(res2, *reinterpret_cast<__m256i*>(b64.values));
    res3 = _mm256_mullo_epi32(res3, *reinterpret_cast<__m256i*>(b64.values));
    res4 = _mm256_mullo_epi32(res4, *reinterpret_cast<__m256i*>(b64.values));
    res5 = _mm256_mullo_epi32(res5, *reinterpret_cast<__m256i*>(b64.values));
    res6 = _mm256_mullo_epi32(res6, *reinterpret_cast<__m256i*>(b64.values));
    res7 = _mm256_mullo_epi32(res7, *reinterpret_cast<__m256i*>(b64.values));
    __m256i z0_7   = _mm256_mullo_epi32(x0_7,   *reinterpret_cast<__m256i*>(p64.values));      // [b^63, .., b^56]
    __m256i z8_15  = _mm256_mullo_epi32(x8_15,  *reinterpret_cast<__m256i*>(p64.values + 8));  // [b^55, .., b^48]
    __m256i z16_23 = _mm256_mullo_epi32(x16_23, *reinterpret_cast<__m256i*>(p64.values + 16)); // [b^47, .., b^40]
    __m256i z24_31 = _mm256_mullo_epi32(x24_31, *reinterpret_cast<__m256i*>(p64.values + 24)); // [b^39, .., b^32]
    __m256i z32_39 = _mm256_mullo_epi32(x32_39, *reinterpret_cast<__m256i*>(p64.values + 32)); // [b^31, .., b^24]
    __m256i z40_47 = _mm256_mullo_epi32(x40_47, *reinterpret_cast<__m256i*>(p64.values + 40)); // [b^23, .., b^16]
    __m256i z48_55 = _mm256_mullo_epi32(x48_55, *reinterpret_cast<__m256i*>(p64.values + 48)); // [b^15, .., b^8 ]
    __m256i z56_63 = _mm256_mullo_epi32(x56_63, *reinterpret_cast<__m256i*>(p64.values + 56)); // [b^7,  .., b, 1]
    res0 = _mm256_add_epi32(res0, z0_7);
    res1 = _mm256_add_epi32(res1, z8_15);
    res2 = _mm256_add_epi32(res2, z16_23);
    res3 = _mm256_add_epi32(res3, z24_31);
    res4 = _mm256_add_epi32(res4, z32_39);
    res5 = _mm256_add_epi32(res5, z40_47);
    res6 = _mm256_add_epi32(res6, z48_55);
    res7 = _mm256_add_epi32(res7, z56_63);

    str += 64;
    n -= 16;
  } while (n >= 16);

  __m256i sum = _mm256_add_epi32(_mm256_add_epi32(squash(res0, res1), squash(res2, res3)),
                                 _mm256_add_epi32(squash(res4, res5), squash(res6, res7)));
  res = _mm_add_epi32(res, _mm256_extracti128_si256(sum, 0));
  res = _mm_add_epi32(res, _mm256_extracti128_si256(sum, 1));
}

inline void polyHashAVX2UnalignedUnroll32(int& n, uint16_t const*& str, __m128i& res) {
  if (n < 8) return;

  res = _mm_mullo_epi32(res, *reinterpret_cast<__m128i*>(b32.values));

  // res0, res1, res2, res3 will accumulate 32 intermediate sums.
  __m256i res0 = _mm256_setzero_si256();
  __m256i res1 = _mm256_setzero_si256();
  __m256i res2 = _mm256_setzero_si256();
  __m256i res3 = _mm256_setzero_si256();

  do {
    __m256i x0_7   = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str));
    __m256i x8_15  = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 8));
    __m256i x16_23 = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 16));
    __m256i x24_31 = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 24));
    res0 = _mm256_mullo_epi32(res0, *reinterpret_cast<__m256i*>(b32.values));
    res1 = _mm256_mullo_epi32(res1, *reinterpret_cast<__m256i*>(b32.values));
    res2 = _mm256_mullo_epi32(res2, *reinterpret_cast<__m256i*>(b32.values));
    res3 = _mm256_mullo_epi32(res3, *reinterpret_cast<__m256i*>(b32.values));
    __m256i z0_7   = _mm256_mullo_epi32(x0_7,   *reinterpret_cast<__m256i*>(p64.values + 32)); // [b^31, .., b^24]
    __m256i z8_15  = _mm256_mullo_epi32(x8_15,  *reinterpret_cast<__m256i*>(p64.values + 40)); // [b^23, .., b^16]
    __m256i z16_23 = _mm256_mullo_epi32(x16_23, *reinterpret_cast<__m256i*>(p64.values + 48)); // [b^15, .., b^8 ]
    __m256i z24_31 = _mm256_mullo_epi32(x24_31, *reinterpret_cast<__m256i*>(p64.values + 56)); // [b^7,  .., b, 1]
    res0 = _mm256_add_epi32(res0, z0_7);
    res1 = _mm256_add_epi32(res1, z8_15);
    res2 = _mm256_add_epi32(res2, z16_23);
    res3 = _mm256_add_epi32(res3, z24_31);

    str += 32;
    n -= 8;
  } while (n >= 8);

  __m256i sum = _mm256_add_epi32(squash(res0, res1), squash(res2, res3));
  res = _mm_add_epi32(res, _mm256_extracti128_si256(sum, 0));
  res = _mm_add_epi32(res, _mm256_extracti128_si256(sum, 1));
}

inline void polyHashAVX2UnalignedUnroll16(int& n, uint16_t const*& str, __m128i& res) {
  if (n < 4) return;

  res = _mm_mullo_epi32(res, *reinterpret_cast<__m128i*>(b16.values));

  // res0, res1 will accumulate 16 intermediate sums.
  __m256i res0 = _mm256_setzero_si256();
  __m256i res1 = _mm256_setzero_si256();

  do {
    __m256i x0_7  = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str));
    __m256i x8_15 = _mm256_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 8));
    res0 = _mm256_mullo_epi32(res0, *reinterpret_cast<__m256i*>(b16.values));
    res1 = _mm256_mullo_epi32(res1, *reinterpret_cast<__m256i*>(b16.values));
    __m256i z0_7  = _mm256_mullo_epi32(x0_7, *reinterpret_cast<__m256i*>(p64.values + 48));  // [b^15, .., b^8]
    __m256i z8_15 = _mm256_mullo_epi32(x8_15, *reinterpret_cast<__m256i*>(p64.values + 56)); // [b^7, .., b, 1]
    res0 = _mm256_add_epi32(res0, z0_7);
    res1 = _mm256_add_epi32(res1, z8_15);

    str += 16;
    n -= 4;
  } while (n >= 4);

  __m256i sum = squash(res0, res1);
  res = _mm_add_epi32(res, _mm256_extracti128_si256(sum, 0));
  res = _mm_add_epi32(res, _mm256_extracti128_si256(sum, 1));
}

inline void polyHashSSEUnalignedTail(int n, uint16_t const* str, __m128i& res) {
  if (n == 0) return;

  __m128i x4_7 = _mm_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str));
  __m128i z4_7 = _mm_mullo_epi32(x4_7, *reinterpret_cast<__m128i*>(p64.values + 60)); // [b^3, b^2, b, 1]

  res = _mm_mullo_epi32(res, *reinterpret_cast<__m128i*>(b4.values));
  __m128i sum = _mm_hadd_epi32(z4_7, z4_7);
  sum = _mm_hadd_epi32(sum, sum);
  res = _mm_add_epi32(res, sum);
}


inline void polyHashAVX2UnalignedTail(int n, uint16_t const* str, __m128i& res) {
  if (n >= 2) {
    __m256i x0_7 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<__m128i const*>(str)));
    __m256i z0_7 = _mm256_mullo_epi32(x0_7, *reinterpret_cast<__m256i*>(p64.values + 56)); // [b^7, .., b, 1]
    res = _mm_mullo_epi32(res, *reinterpret_cast<__m128i*>(b8.values));
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

  polyHashAVX2UnalignedUnroll16(n, str, res);
  polyHashAVX2UnalignedTail(n, str, res);

  return _mm_cvtsi128_si32(res);
}

int polyHashAVX2UnalignedUnrollUpTo32(int n, uint16_t const* str) {
  __m128i res = _mm_setzero_si128();

  polyHashAVX2UnalignedUnroll32(n, str, res);
  polyHashAVX2UnalignedUnroll16(n, str, res);
  polyHashAVX2UnalignedTail(n, str, res);

  return _mm_cvtsi128_si32(res);
}

int polyHashAVX2UnalignedUnrollUpTo64(int n, uint16_t const* str) {
  __m128i res = _mm_setzero_si128();

  polyHashAVX2UnalignedUnroll64(n, str, res);
  polyHashAVX2UnalignedUnroll32(n, str, res);
  polyHashAVX2UnalignedUnroll16(n, str, res);
  polyHashAVX2UnalignedTail(n, str, res);

  return _mm_cvtsi128_si32(res);
}

inline void polyHashSSEUnalignedUnroll16(int& n, uint16_t const*& str, __m128i& res) {
  if (n < 4) return;

  // res0, res1, res2, res3 will accumulate 16 intermediate sums.
  __m128i res0 = _mm_setzero_si128();
  __m128i res1 = _mm_setzero_si128();
  __m128i res2 = _mm_setzero_si128();
  __m128i res3 = _mm_setzero_si128();

  do {
    __m128i x0_7   = _mm_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str));
    __m128i x8_15  = _mm_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 4));
    __m128i x16_23 = _mm_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 8));
    __m128i x24_31 = _mm_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 12));
    res0 = _mm_mullo_epi32(res0, *reinterpret_cast<__m128i*>(b16.values));
    res1 = _mm_mullo_epi32(res1, *reinterpret_cast<__m128i*>(b16.values));
    res2 = _mm_mullo_epi32(res2, *reinterpret_cast<__m128i*>(b16.values));
    res3 = _mm_mullo_epi32(res3, *reinterpret_cast<__m128i*>(b16.values));
    __m128i z0_7   = _mm_mullo_epi32(x0_7,   *reinterpret_cast<__m128i*>(p64.values + 48)); // [b^15, b^14, b^13, b^12]
    __m128i z8_15  = _mm_mullo_epi32(x8_15,  *reinterpret_cast<__m128i*>(p64.values + 52)); // [b^11, b^10, b^9,  b^8 ]
    __m128i z16_23 = _mm_mullo_epi32(x16_23, *reinterpret_cast<__m128i*>(p64.values + 56)); // [b^7,  b^6,  b^5,  b^4 ]
    __m128i z24_31 = _mm_mullo_epi32(x24_31, *reinterpret_cast<__m128i*>(p64.values + 60)); // [b^3,  b^2,  b,    1   ]
    res0 = _mm_add_epi32(res0, z0_7);
    res1 = _mm_add_epi32(res1, z8_15);
    res2 = _mm_add_epi32(res2, z16_23);
    res3 = _mm_add_epi32(res3, z24_31);

    str += 16;
    n -= 4;
  } while (n >= 4);

  res = _mm_add_epi32(res, _mm_add_epi32(squash(res0, res1), squash(res2, res3)));
}

inline void polyHashSSEUnalignedUnroll8(int& n, uint16_t const*& str, __m128i& res) {
  if (n < 2) return;

  res = _mm_mullo_epi32(res, *reinterpret_cast<__m128i*>(b8.values));

  // res0, res1 will accumulate 8 intermediate sums.
  __m128i res0 = _mm_setzero_si128();
  __m128i res1 = _mm_setzero_si128();

  do {
    __m128i x0_7  = _mm_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str));
    __m128i x8_15 = _mm_cvtepu16_epi32(*reinterpret_cast<__m128i const*>(str + 4));
    res0 = _mm_mullo_epi32(res0, *reinterpret_cast<__m128i*>(b8.values));
    res1 = _mm_mullo_epi32(res1, *reinterpret_cast<__m128i*>(b8.values));
    __m128i z0_7  = _mm_mullo_epi32(x0_7,  *reinterpret_cast<__m128i*>(p64.values + 56)); // [b^7, b^6, b^5, b^4]
    __m128i z8_15 = _mm_mullo_epi32(x8_15, *reinterpret_cast<__m128i*>(p64.values + 60)); // [b^3, b^2, b,   1  ]
    res0 = _mm_add_epi32(res0, z0_7);
    res1 = _mm_add_epi32(res1, z8_15);

    str += 8;
    n -= 2;
  } while (n >= 2);

  res = _mm_add_epi32(res, squash(res0, res1));
}

int polyHashSSEUnalignedUnrollUpTo8(int n, uint16_t const* str) {
  __m128i res = _mm_setzero_si128();

  polyHashSSEUnalignedUnroll8(n, str, res);
  polyHashSSEUnalignedTail(n, str, res);

  return _mm_cvtsi128_si32(res);
}

int polyHashSSEUnalignedUnrollUpTo16(int n, uint16_t const* str) {
  __m128i res = _mm_setzero_si128();

  polyHashSSEUnalignedUnroll16(n, str, res);
  polyHashSSEUnalignedUnroll8(n, str, res);
  polyHashSSEUnalignedTail(n, str, res);

  return _mm_cvtsi128_si32(res);
}

#if defined(__x86_64__)
  const bool x64 = true;
#else
  const bool x64 = false;
#endif
  const bool sseSupported = __builtin_cpu_supports("sse4.1");
  const bool avx2Supported = __builtin_cpu_supports("avx2");

}

int polyHash_x86(int length, uint16_t const* str) {
  if (length < 20 || (!sseSupported && !avx2Supported)) {
    // Either vectorization is not supported or the string is too short to gain from it.
    int res = 0;
    for (int i = 0; i < length; ++i)
      res = res * 31 + str[i];
    return res;
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

#endif
