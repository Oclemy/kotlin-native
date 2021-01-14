/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#include "polyhash/PolyHash.h"

int polyHash_x86(int length, int16_t const* str);
int polyHash_arm(int length, int16_t const* str);

int polyHash(int length, int16_t const* str) {
#if defined(__x86_64__) or defined(__i386__)
  return polyHash_x86(length, str);
#elif defined(__arm__) or defined(__aarch64__)
  return polyHash_arm(length, str);
#else
  // Default naive impl.
  int res = 0;
  for (int i = 0; i < length; ++i)
    res = res * 31 + str[i];
  return res;
#endif
}