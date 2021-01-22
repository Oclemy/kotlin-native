/*
 * Copyright 2010-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

#include "Freezing.hpp"

#include <deque>
#include <vector>

#include "ExtraObjectData.hpp"
#include "FreezeHooks.hpp"
#include "Memory.h"
#include "Natives.h"
#include "Types.h"

using namespace kotlin;

namespace {

// TODO: Come up with a better way to iterate object fields.
template <typename func>
inline void traverseObjectFields(ObjHeader* obj, func process) {
    const TypeInfo* typeInfo = obj->type_info();
    if (typeInfo != theArrayTypeInfo) {
        for (int index = 0; index < typeInfo->objOffsetsCount_; index++) {
            ObjHeader** location = reinterpret_cast<ObjHeader**>(reinterpret_cast<uintptr_t>(obj) + typeInfo->objOffsets_[index]);
            process(*location);
        }
    } else {
        ArrayHeader* array = obj->array();
        for (uint32_t index = 0; index < array->count_; index++) {
            process(*ArrayAddressOfElementAt(array, index));
        }
    }
}

// `func` is called in a preorder fashion, and so is allowed to modify subgraph of the object passed to it.
template <typename Func>
void dfs(ObjHeader* root, Func func) noexcept {
    std::deque<ObjHeader*> queue;
    queue.push_back(root);
    while (!queue.empty()) {
        ObjHeader* object = queue.front();
        queue.pop_front();
        func(object);
        traverseObjectFields(object, [&queue](ObjHeader* field) noexcept { queue.push_back(field); });
    }
}

} // namespace

ObjHeader* mm::FreezeSubgraph(ObjHeader* object) noexcept {
    std::vector<ObjHeader*> objects;
    dfs(object, [&objects](ObjHeader* object) noexcept {
        objects.push_back(object);
        RunFreezeHooks(object);
    });
    for (auto* object : objects) {
        if (auto* metaObject = object->GetMetaObjHeader()) {
            auto& extraObjectData = mm::ExtraObjectData::FromMetaObjHeader(metaObject);
            if (!extraObjectData.CanBeFrozen()) return object;
        }
    }
    for (auto* object : objects) {
        auto& extraObjectData = mm::ExtraObjectData::FromMetaObjHeader(object->meta_object());
        extraObjectData.Freeze();
    }
    return nullptr;
}
