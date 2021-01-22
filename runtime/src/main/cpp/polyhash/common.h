/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#ifndef RUNTIME_POLYHASH_COMMON_H
#define RUNTIME_POLYHASH_COMMON_H

#include <array>
#include <cstdint>
#include "polyhash/naive.h"

constexpr uint32_t Power(uint32_t base, uint8_t exponent) {
    uint32_t result = 1;
    for (uint8_t i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

template <uint8_t Exponent>
constexpr std::array<uint32_t, Exponent> DecreasingPowers(uint32_t base) {
    std::array<uint32_t, Exponent> result = {};
    uint32_t current = 1;
    for (auto it = result.rbegin(); it != result.rend(); ++it) {
        *it = current;
        current *= base;
    }
    return result;
}

template <size_t Count>
constexpr std::array<uint32_t, Count> RepeatingPowers(uint32_t base, uint8_t exponent) {
    std::array<uint32_t, Count> result = {};
    uint32_t value = Power(base, exponent);
    for (auto& element : result)
        element = value;
    return result;
}

#define polyHashUnroll2(n, str, res, vecType, vec128Type, u16VecType, b, p, pShift, initVec, u16Load,      \
                        vecMul, vecAdd, vec128Mul, vec128Add, squash) do {                                 \
    const int vecLength = sizeof(vecType) / 4;                                                             \
    if (n >= vecLength / 2) {                                                                              \
        res = vec128Mul(res, *reinterpret_cast<vec128Type const*>(&b[0]));                                 \
                                                                                                           \
        vecType res0 = initVec;                                                                            \
        vecType res1 = initVec;                                                                            \
                                                                                                           \
        do {                                                                                               \
            vecType x0 = u16Load(*reinterpret_cast<u16VecType const*>(str));                               \
            vecType x1 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength));                   \
            res0 = vecMul(res0, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res1 = vecMul(res1, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            vecType z0 = vecMul(x0, *reinterpret_cast<vecType const*>(&p[pShift]));                        \
            vecType z1 = vecMul(x1, *reinterpret_cast<vecType const*>(&p[pShift + vecLength]));            \
            res0 = vecAdd(res0, z0);                                                                       \
            res1 = vecAdd(res1, z1);                                                                       \
                                                                                                           \
            str += vecLength * 2;                                                                          \
            n -= vecLength / 2;                                                                            \
        } while (n >= vecLength / 2);                                                                      \
                                                                                                           \
        res = vec128Add(res, squash(res0, res1));                                                          \
    }                                                                                                      \
} while(0)

#define polyHashUnroll4(n, str, res, vecType, vec128Type, u16VecType, b, p, pShift, initVec, u16Load,      \
                        vecMul, vecAdd, vec128Mul, vec128Add, squash) do {                                 \
    const int vecLength = sizeof(vecType) / 4;                                                             \
    if (n >= vecLength) {                                                                                  \
        res = vec128Mul(res, *reinterpret_cast<vec128Type const*>(&b[0]));                                 \
                                                                                                           \
        vecType res0 = initVec;                                                                            \
        vecType res1 = initVec;                                                                            \
        vecType res2 = initVec;                                                                            \
        vecType res3 = initVec;                                                                            \
                                                                                                           \
        do {                                                                                               \
            vecType x0 = u16Load(*reinterpret_cast<u16VecType const*>(str));                               \
            vecType x1 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength));                   \
            vecType x2 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength * 2));               \
            vecType x3 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength * 3));               \
            res0 = vecMul(res0, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res1 = vecMul(res1, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res2 = vecMul(res2, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res3 = vecMul(res3, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            vecType z0 = vecMul(x0, *reinterpret_cast<vecType const*>(&p[pShift]));                        \
            vecType z1 = vecMul(x1, *reinterpret_cast<vecType const*>(&p[pShift + vecLength]));            \
            vecType z2 = vecMul(x2, *reinterpret_cast<vecType const*>(&p[pShift + vecLength * 2]));        \
            vecType z3 = vecMul(x3, *reinterpret_cast<vecType const*>(&p[pShift + vecLength * 3]));        \
            res0 = vecAdd(res0, z0);                                                                       \
            res1 = vecAdd(res1, z1);                                                                       \
            res2 = vecAdd(res2, z2);                                                                       \
            res3 = vecAdd(res3, z3);                                                                       \
                                                                                                           \
            str += vecLength * 4;                                                                          \
            n -= vecLength;                                                                                \
        } while (n >= vecLength);                                                                          \
                                                                                                           \
        res = vec128Add(res, vec128Add(squash(res0, res1), squash(res2, res3)));                           \
    }                                                                                                      \
} while(0)

#define polyHashUnroll8(n, str, res, vecType, u16VecType, b, p, pShift, initVec, u16Load,                  \
                        vecMul, vecAdd, vec128Add, squash) do {                                            \
    const int vecLength = sizeof(vecType) / 4;                                                             \
    if (n >= vecLength * 2) {                                                                              \
        vecType res0 = initVec;                                                                            \
        vecType res1 = initVec;                                                                            \
        vecType res2 = initVec;                                                                            \
        vecType res3 = initVec;                                                                            \
        vecType res4 = initVec;                                                                            \
        vecType res5 = initVec;                                                                            \
        vecType res6 = initVec;                                                                            \
        vecType res7 = initVec;                                                                            \
                                                                                                           \
        do {                                                                                               \
            vecType x0 = u16Load(*reinterpret_cast<u16VecType const*>(str));                               \
            vecType x1 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength));                   \
            vecType x2 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength * 2));               \
            vecType x3 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength * 3));               \
            vecType x4 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength * 4));               \
            vecType x5 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength * 5));               \
            vecType x6 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength * 6));               \
            vecType x7 = u16Load(*reinterpret_cast<u16VecType const*>(str + vecLength * 7));               \
            res0 = vecMul(res0, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res1 = vecMul(res1, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res2 = vecMul(res2, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res3 = vecMul(res3, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res4 = vecMul(res4, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res5 = vecMul(res5, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res6 = vecMul(res6, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            res7 = vecMul(res7, *reinterpret_cast<vecType const*>(&b[0]));                                 \
            vecType z0 = vecMul(x0, *reinterpret_cast<vecType const*>(&p[pShift]));                        \
            vecType z1 = vecMul(x1, *reinterpret_cast<vecType const*>(&p[pShift + vecLength]));            \
            vecType z2 = vecMul(x2, *reinterpret_cast<vecType const*>(&p[pShift + vecLength * 2]));        \
            vecType z3 = vecMul(x3, *reinterpret_cast<vecType const*>(&p[pShift + vecLength * 3]));        \
            vecType z4 = vecMul(x4, *reinterpret_cast<vecType const*>(&p[pShift + vecLength * 4]));        \
            vecType z5 = vecMul(x5, *reinterpret_cast<vecType const*>(&p[pShift + vecLength * 5]));        \
            vecType z6 = vecMul(x6, *reinterpret_cast<vecType const*>(&p[pShift + vecLength * 6]));        \
            vecType z7 = vecMul(x7, *reinterpret_cast<vecType const*>(&p[pShift + vecLength * 7]));        \
            res0 = vecAdd(res0, z0);                                                                       \
            res1 = vecAdd(res1, z1);                                                                       \
            res2 = vecAdd(res2, z2);                                                                       \
            res3 = vecAdd(res3, z3);                                                                       \
            res4 = vecAdd(res4, z4);                                                                       \
            res5 = vecAdd(res5, z5);                                                                       \
            res6 = vecAdd(res6, z6);                                                                       \
            res7 = vecAdd(res7, z7);                                                                       \
                                                                                                           \
            str += vecLength * 8;                                                                          \
            n -= vecLength * 2;                                                                            \
        } while (n >= vecLength * 2);                                                                      \
                                                                                                           \
        res = vec128Add(res, vec128Add(vec128Add(squash(res0, res1), squash(res2, res3)),                  \
                                       vec128Add(squash(res4, res5), squash(res6, res7))));                \
    }                                                                                                      \
} while(0)

#endif  // RUNTIME_POLYHASH_COMMON_H
