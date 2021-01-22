/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#include "FreezeHooks.hpp"

#include "Memory.h"
#include "Types.h"
#include "WorkerBoundReference.h"

using namespace kotlin;

void kotlin::RunFreezeHooks(ObjHeader* object) noexcept {
    if (object->type_info() == theWorkerBoundReferenceTypeInfo) {
        WorkerBoundReferenceFreezeHook(object);
    }
}
