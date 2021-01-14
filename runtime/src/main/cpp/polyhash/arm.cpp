/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#include "polyhash/common.h"

#if defined(__arm__) or defined(__aarch64__)

// TODO: Vectorize.
int polyHash_arm(int length, int16_t const* str) {
  int res = 0;
  for (int i = 0; i < length; ++i)
    res = res * 31 + str[i];
  return res;
}

#endif