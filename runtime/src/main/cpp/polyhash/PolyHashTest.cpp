/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#include "polyhash/PolyHash.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

int polyHashNaive(int length, int16_t* str) {
  int res = 0;
  for (int i = 0; i < length; ++i)
    res = res * 31 + str[i];
  return res;
}

TEST(PolyHashTest, Correctness) {
  const int maxLength = 10000;
  int16_t str[maxLength + 1];
  for (int k = 1; k <= 10000; ++k) {
    for (int i = 0; i < k; ++i)
      str[i] = k + i;
    str[k] = 0;

    EXPECT_EQ(polyHashNaive(k, str), polyHash(k, str));
  }
}

}