/*
 * Copyright 2010-2018 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license
 * that can be found in the LICENSE file.
 */

package kotlin.native.concurrent

import kotlin.native.identityHashCode
import kotlin.reflect.KClass
import kotlinx.cinterop.*
import kotlinx.cinterop.NativePtr
import kotlin.native.internal.*
import kotlin.native.internal.DescribeObjectForDebugging
import kotlin.native.internal.GCCritical
import kotlin.native.internal.InternalForKotlinNative
import kotlin.native.internal.debugDescription

// Implementation details.

@SymbolName("Kotlin_Worker_stateOfFuture")
@GCCritical // locks, no allocations.
external internal fun stateOfFuture(id: Int): Int

@SymbolName("Kotlin_Worker_consumeFuture")
@GCCritical
@PublishedApi
external internal fun consumeFuture(id: Int): Any?

@SymbolName("Kotlin_Worker_waitForAnyFuture")
// TODO: Lock + condvar wait. no allocs.
external internal fun waitForAnyFuture(versionToken: Int, millis: Int): Boolean

@SymbolName("Kotlin_Worker_versionToken")
// TODO: Lock + no allocs. Important: Locks have fast paths. Do we need it?
external internal fun versionToken(): Int

@kotlin.native.internal.ExportForCompiler
internal fun executeImpl(worker: Worker, mode: TransferMode, producer: () -> Any?,
                         job: CPointer<CFunction<*>>): Future<Any?> =
        Future<Any?>(executeInternal(worker.id, mode.value, producer, job))

@SymbolName("Kotlin_Worker_startInternal")
// TODO: Lock + allocs.
external internal fun startInternal(errorReporting: Boolean, name: String?): Int

@SymbolName("Kotlin_Worker_currentInternal")
@GCCritical // No locks, no allocations
external internal fun currentInternal(): Int

@SymbolName("Kotlin_Worker_requestTerminationWorkerInternal")
// TODO: Locks and allocs.
external internal fun requestTerminationInternal(id: Int, processScheduledJobs: Boolean): Int

@SymbolName("Kotlin_Worker_executeInternal")
external internal fun executeInternal(
        id: Int, mode: Int, producer: () -> Any?, job: CPointer<CFunction<*>>): Int

@SymbolName("Kotlin_Worker_executeAfterInternal")
external internal fun executeAfterInternal(id: Int, operation: () -> Unit, afterMicroseconds: Long): Unit

@SymbolName("Kotlin_Worker_processQueueInternal")
external internal fun processQueueInternal(id: Int): Boolean

@SymbolName("Kotlin_Worker_parkInternal")
external internal fun parkInternal(id: Int, timeoutMicroseconds: Long, process: Boolean): Boolean

@SymbolName("Kotlin_Worker_getNameInternal")
external internal fun getWorkerNameInternal(id: Int): String?

@ExportForCppRuntime
internal fun ThrowWorkerUnsupported(): Unit =
        throw UnsupportedOperationException("Workers are not supported")

@ExportForCppRuntime
internal fun ThrowWorkerInvalidState(): Unit =
        throw IllegalStateException("Illegal transfer state")

@ExportForCppRuntime
internal fun WorkerLaunchpad(function: () -> Any?) = function()

@PublishedApi
@SymbolName("Kotlin_Worker_detachObjectGraphInternal")
@GCCritical
external internal fun detachObjectGraphInternal(mode: Int, producer: () -> Any?): NativePtr

@PublishedApi
@SymbolName("Kotlin_Worker_attachObjectGraphInternal")
@GCCritical
external internal fun attachObjectGraphInternal(stable: NativePtr): Any?

@SymbolName("Kotlin_Worker_freezeInternal")
@GCCritical
internal external fun freezeInternal(it: Any?)

@SymbolName("Kotlin_Worker_isFrozenInternal")
internal external fun isFrozenInternal(it: Any?): Boolean

@ExportForCppRuntime
internal fun ThrowFreezingException(toFreeze: Any, blocker: Any): Nothing =
        throw FreezingException(toFreeze, blocker)

@ExportForCppRuntime
internal fun ThrowInvalidMutabilityException(where: Any): Nothing {
    val description = debugDescription(where::class, where.identityHashCode())
    throw InvalidMutabilityException("mutation attempt of frozen $description")
}

@ExportForCppRuntime
internal fun ThrowIllegalObjectSharingException(typeInfo: NativePtr, address: NativePtr) {
    val description = DescribeObjectForDebugging(typeInfo, address)
    throw IncorrectDereferenceException("illegal attempt to access non-shared $description from other thread")
}

@SymbolName("Kotlin_AtomicReference_checkIfFrozen")
external internal fun checkIfFrozen(ref: Any?)

@InternalForKotlinNative
@SymbolName("Kotlin_Worker_waitTermination")
external public fun waitWorkerTermination(worker: Worker)
