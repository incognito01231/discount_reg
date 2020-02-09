/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_ray_runtime_context_NativeWorkerContext */

#ifndef _Included_org_ray_runtime_context_NativeWorkerContext
#define _Included_org_ray_runtime_context_NativeWorkerContext
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_ray_runtime_context_NativeWorkerContext
 * Method:    nativeGetCurrentTaskType
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_org_ray_runtime_context_NativeWorkerContext_nativeGetCurrentTaskType(JNIEnv *,
                                                                          jclass, jlong);

/*
 * Class:     org_ray_runtime_context_NativeWorkerContext
 * Method:    nativeGetCurrentTaskId
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL
Java_org_ray_runtime_context_NativeWorkerContext_nativeGetCurrentTaskId(JNIEnv *, jclass,
                                                                        jlong);

/*
 * Class:     org_ray_runtime_context_NativeWorkerContext
 * Method:    nativeGetCurrentJobId
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL
Java_org_ray_runtime_context_NativeWorkerContext_nativeGetCurrentJobId(JNIEnv *, jclass,
                                                                       jlong);

/*
 * Class:     org_ray_runtime_context_NativeWorkerContext
 * Method:    nativeGetCurrentWorkerId
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL
Java_org_ray_runtime_context_NativeWorkerContext_nativeGetCurrentWorkerId(JNIEnv *,
                                                                          jclass, jlong);

/*
 * Class:     org_ray_runtime_context_NativeWorkerContext
 * Method:    nativeGetCurrentActorId
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL
Java_org_ray_runtime_context_NativeWorkerContext_nativeGetCurrentActorId(JNIEnv *, jclass,
                                                                         jlong);

#ifdef __cplusplus
}
#endif
#endif
