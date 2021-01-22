/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#ifndef RUNTIME_MM_FREEZING_H
#define RUNTIME_MM_FREEZING_H

struct ObjHeader;

namespace kotlin {
namespace mm {

// If some object in the `object` subgraph was marked with `EnsureNeverFrozen` only
// freeze hooks would've been executed. On success, returns `nullptr`.
// Note: not thread safe.
ObjHeader* FreezeSubgraph(ObjHeader* object) noexcept;

} // namespace mm
} // namespace kotlin

#endif // RUNTIME_MM_FREEZING_H
