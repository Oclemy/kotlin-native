/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#ifndef RUNTIME_POLYHASH_COMMON_H
#define RUNTIME_POLYHASH_COMMON_H

#include <stdint.h>

template <int... Values>
struct IntList;

template<>
struct IntList<> {};

template<int... Values>
struct Length;

template<int H, int... T>
struct Length<H, T...> {
    static const int value = 1 + Length<T...>::value;
};

template<>
struct Length<> {
    static const int value = 0;
};

template<typename IL1, typename IL2>
struct Concat;

template<int... Is1, int... Is2>
struct Concat<IntList<Is1...>, IntList<Is2...>> {
    using type = IntList<Is1..., Is2...>;
};

template<int N>
struct Descent {
    using type = typename Concat<IntList<N - 1>, typename Descent<N - 1>::type>::type;
};

template<>
struct Descent<0> {
    using type = IntList<>;
};

template<int N, int K>
struct Repeat {
    using type = typename Concat<IntList<K>, typename Repeat<N - 1, K>::type>::type;
};

template<int K>
struct Repeat<0, K> {
    using type = IntList<>;
};

template<int B, int K>
struct Power {
    static const int value = (Power<B, K - 1>::value * static_cast<int64_t>(B)) & 0xFFFFFFFF;
};

template<int B>
struct Power<B, 0> {
    static const int value = 1;
};

template<int B, typename T>
struct Powers;

template<int B, int... Values>
struct Powers<B, IntList<Values...>> {
    __attribute__((aligned(32))) int values[Length<Values...>::value] = { Power<B, Values>::value... };
};

template<int B, int K>
struct DecreasingPowers : Powers<B, typename Descent<K>::type> { };

template<int B, int N, int K>
struct RepeatingPowers : Powers<B, typename Repeat<N, K>::type> { };

#endif  // RUNTIME_POLYHASH_COMMON_H
