/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 *
 *
 * Custom Storage Checkpoint and Restore Demo
 * Requires display driver 615 or higher and CUDA Toolkit 13.4 or higher
 *
 * Usage: r615-custom-storage [checkpoint-file]
 *
 * A parent process launches a child. The child initializes CUDA and allocates
 * GPU memory. The parent then:
 *   1. Locks and checkpoints the child via the CUDA Driver custom-storage API
 *   2. Copies the mapped GPU memory to a file
 *   3. Restores GPU memory from the file via the custom-storage API
 *   4. Unlocks the child so it can verify GPU contents survived
 *
 * The checkpoint file defaults to /tmp/cuda-checkpoint-custom-storage.bin and
 * is left in place after the run.
 *
 * A complete checkpoint would also dump and restore the child's CPU and host
 * state with a utility such as CRIU, between steps 2 and 3. This sample leaves
 * that out to keep the focus on the custom-storage API.
 *
 * The parent must cuInit and retain the primary context of each device the
 * child uses. Checkpointing the calling process to custom storage is not
 * supported.
 *
 * Build with CUDA Toolkit 13.4 or higher, whose cuda.h declares the
 * custom-storage Driver API:
 * gcc -I /usr/local/cuda/include r615-custom-storage.c -o r615-custom-storage -lcuda
 */

 #include <stdlib.h>
 #include <stdio.h>
 #include <stdbool.h>
 
 #include <sys/types.h>
 #include <sys/socket.h>
 #include <sys/wait.h>
 #include <unistd.h>
 
 #include <cuda.h>
 
 #define CHECK(x) check((x) != 0, #x, __FILE__, __LINE__)
 #define CHECK_EQ(x, y) CHECK((x) == (y))
 #define CHECK_OK(x) CHECK_EQ(x, 0)
 
 #define MAX_DEVICES 32
 #define ALLOC_SIZE (4 * 1024 * 1024)
 #define MAGIC_BYTE 0xCD
 #define DEFAULT_MEM_FILE "/tmp/cuda-checkpoint-custom-storage.bin"
 
 void check(bool is_ok, const char *msg, const char *file, int line)
 {
     if (!is_ok) {
         fprintf(stderr, "Error: \"%s\" failed at %s:%d\n", msg, file, line);
         abort();
     }
 }
 
 void runParent(pid_t child, int sock, const char *memFilePath);
 void checkpointAndRestore(pid_t child, const char *memFilePath);
 void checkpointToFile(pid_t child, const char *memFilePath);
 void restoreFromFile(pid_t child, const char *memFilePath);
 void runChild(int sock);
 void expectState(pid_t pid, CUprocessState expected);
 
 static char scratch = '\0';
 
 int main(int argc, char **argv)
 {
     const char *memFilePath = argc > 1 ? argv[1] : DEFAULT_MEM_FILE;
     int socks[2];
     CHECK_OK(socketpair(AF_UNIX, SOCK_STREAM, 0, socks));
 
     pid_t child = fork();
     if (child == 0) {
         CHECK_OK(close(socks[1]));
         runChild(socks[0]);
         exit(0);
     }
 
     CHECK_OK(close(socks[0]));
     runParent(child, socks[1], memFilePath);
     return 0;
 }
 
 void expectState(pid_t pid, CUprocessState expected)
 {
     CUprocessState state;
     CHECK_OK(cuCheckpointProcessGetState(pid, &state));
     CHECK_EQ(state, expected);
 }
 
 void runParent(pid_t child, int sock, const char *memFilePath)
 {
     // wait for checkpoint request
     CHECK_EQ(read(sock, &scratch, 1), 1);
 
     expectState(child, CU_PROCESS_STATE_RUNNING);
     checkpointAndRestore(child, memFilePath);
 
     // unblock child upon restore
     CHECK_EQ(write(sock, &scratch, 1), 1);
 
     int status = 0;
     CHECK_EQ(waitpid(child, &status, 0), child);
     CHECK(WIFEXITED(status));
     CHECK_OK(WEXITSTATUS(status));
 }
 
 void checkpointAndRestore(pid_t child, const char *memFilePath)
 {
     int devCount = 0;
 
     CHECK_OK(cuInit(0));
 
     // custom storage requires the caller to retain the primary context of every device
     CHECK_OK(cuDeviceGetCount(&devCount));
     for (int i = 0; i < devCount; i++) {
         CUdevice dev = 0;
         CUcontext ctx = NULL;
         CHECK_OK(cuDeviceGet(&dev, i));
         CHECK_OK(cuDevicePrimaryCtxRetain(&ctx, dev));
     }
 
     checkpointToFile(child, memFilePath);
 
     // The child's GPU contents are now in memFilePath and the child holds no GPU
     // resources. A complete checkpoint would dump the child's CPU and host state
     // here with a utility such as CRIU, and restore it before the GPU memory is
     // restored below.
 
     restoreFromFile(child, memFilePath);
 }
 
 void checkpointToFile(pid_t child, const char *memFilePath)
 {
     void *hostPtrs[MAX_DEVICES] = {0};
     CUcontext ctxs[MAX_DEVICES] = {0};
 
     CUcheckpointCustomStorageInfo *checkpointInfo = NULL;
     CUcheckpointCheckpointArgs checkpointArgs = {0};
     checkpointArgs.customStorageInfo_out = &checkpointInfo;
 
     CHECK_OK(cuCheckpointProcessLock(child, NULL));
     expectState(child, CU_PROCESS_STATE_LOCKED);
 
     CHECK_OK(cuCheckpointProcessCheckpoint(child, &checkpointArgs));
     expectState(child, CU_PROCESS_STATE_CHECKPOINTING);
     CHECK(checkpointInfo != NULL);
     CHECK(checkpointInfo->deviceCount <= MAX_DEVICES);
 
     for (unsigned i = 0; i < checkpointInfo->deviceCount; i++) {
         CUcheckpointCustomStoragePerDeviceData *gpu = &checkpointInfo->perDeviceData[i];
         CHECK_OK(cuStreamGetCtx(gpu->stream, &ctxs[i]));
         CHECK_OK(cuCtxSetCurrent(ctxs[i]));
         CHECK_OK(cuMemAllocHost(&hostPtrs[i], gpu->size));
         CHECK_OK(cuMemcpyDtoHAsync(hostPtrs[i], gpu->devPtr, gpu->size, gpu->stream));
         CHECK_OK(cuStreamSynchronize(gpu->stream));
     }
 
     FILE *file = fopen(memFilePath, "wb");
     CHECK(file != NULL);
     for (unsigned i = 0; i < checkpointInfo->deviceCount; i++) {
         CHECK_EQ(fwrite(hostPtrs[i], 1, checkpointInfo->perDeviceData[i].size, file),
                  checkpointInfo->perDeviceData[i].size);
     }
     CHECK_OK(fclose(file));
 
     for (unsigned i = 0; i < checkpointInfo->deviceCount; i++) {
         CHECK_OK(cuMemFreeHost(hostPtrs[i]));
     }
 
     CHECK_OK(cuCheckpointOperationComplete(checkpointInfo->handle));
     expectState(child, CU_PROCESS_STATE_CHECKPOINTED);
 
     printf("checkpointed GPU contents of %d to %s\n", child, memFilePath);
 }
 
 void restoreFromFile(pid_t child, const char *memFilePath)
 {
     void *hostPtrs[MAX_DEVICES] = {0};
     CUcontext ctxs[MAX_DEVICES] = {0};
 
     CUcheckpointCustomStorageInfo *restoreInfo = NULL;
     CUcheckpointRestoreArgs restoreArgs = {0};
     restoreArgs.customStorageInfo_out = &restoreInfo;
 
     CHECK_OK(cuCheckpointProcessRestore(child, &restoreArgs));
     expectState(child, CU_PROCESS_STATE_RESTORING);
     CHECK(restoreInfo != NULL);
     CHECK(restoreInfo->deviceCount <= MAX_DEVICES);
 
     for (unsigned i = 0; i < restoreInfo->deviceCount; i++) {
         CUcheckpointCustomStoragePerDeviceData *gpu = &restoreInfo->perDeviceData[i];
         CHECK_OK(cuStreamGetCtx(gpu->stream, &ctxs[i]));
         CHECK_OK(cuCtxSetCurrent(ctxs[i]));
         CHECK_OK(cuMemAllocHost(&hostPtrs[i], gpu->size));
     }
 
     FILE *file = fopen(memFilePath, "rb");
     CHECK(file != NULL);
     for (unsigned i = 0; i < restoreInfo->deviceCount; i++) {
         CHECK_EQ(fread(hostPtrs[i], 1, restoreInfo->perDeviceData[i].size, file),
                  restoreInfo->perDeviceData[i].size);
     }
     CHECK_OK(fclose(file));
 
     for (unsigned i = 0; i < restoreInfo->deviceCount; i++) {
         CUcheckpointCustomStoragePerDeviceData *gpu = &restoreInfo->perDeviceData[i];
         CHECK_OK(cuCtxSetCurrent(ctxs[i]));
         CHECK_OK(cuMemcpyHtoDAsync(gpu->devPtr, hostPtrs[i], gpu->size, gpu->stream));
     }
 
     CHECK_OK(cuCheckpointOperationComplete(restoreInfo->handle));
     expectState(child, CU_PROCESS_STATE_LOCKED);
 
     for (unsigned i = 0; i < restoreInfo->deviceCount; i++) {
         CHECK_OK(cuMemFreeHost(hostPtrs[i]));
     }
 
     CHECK_OK(cuCheckpointProcessUnlock(child, NULL));
     expectState(child, CU_PROCESS_STATE_RUNNING);
 }
 
 void runChild(int sock)
 {
     int devCount = 0;
     CUdeviceptr ptrs[MAX_DEVICES] = {0};
 
     CHECK_OK(cuInit(0));
     CHECK_OK(cuDeviceGetCount(&devCount));
     CHECK(devCount <= MAX_DEVICES);
 
     for (int i = 0; i < devCount; i++) {
         CUdevice dev = 0;
         CUcontext ctx = NULL;
         CHECK_OK(cuDeviceGet(&dev, i));
         CHECK_OK(cuDevicePrimaryCtxRetain(&ctx, dev));
         CHECK_OK(cuCtxSetCurrent(ctx));
         CHECK_OK(cuMemAlloc(&ptrs[i], ALLOC_SIZE));
         CHECK_OK(cuMemsetD8(ptrs[i], MAGIC_BYTE, ALLOC_SIZE));
         CHECK_OK(cuCtxSynchronize());
     }
 
     // request checkpoint and wait for restore
     CHECK_EQ(write(sock, &scratch, 1), 1);
     CHECK_EQ(read(sock, &scratch, 1), 1);
 
     unsigned char *host = malloc(ALLOC_SIZE);
     CHECK(host != NULL);
     for (int i = 0; i < devCount; i++) {
         CUdevice dev = 0;
         CUcontext ctx = NULL;
         CHECK_OK(cuDeviceGet(&dev, i));
         CHECK_OK(cuDevicePrimaryCtxRetain(&ctx, dev));
         CHECK_OK(cuCtxSetCurrent(ctx));
         CHECK_OK(cuMemcpyDtoH(host, ptrs[i], ALLOC_SIZE));
         for (size_t b = 0; b < ALLOC_SIZE; b++) {
             CHECK_EQ(host[b], MAGIC_BYTE);
         }
         CHECK_OK(cuMemFree(ptrs[i]));
     }
     free(host);
 
     printf("SUCCESS\n");
 }
 