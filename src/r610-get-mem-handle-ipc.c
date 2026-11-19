/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Checkpoint and Restore cuIpcGetMemHandle Demo
 * Requires display driver 610 or higher
 *
 * Build with the CUDA toolkit as follows:
 * gcc -I /usr/local/cuda/include -pthread r610-get-mem-handle-ipc.c -o r610-get-mem-handle-ipc -lcuda
 */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include <pthread.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda.h>

#define CHECK(x) check((x) != 0, #x, __FILE__, __LINE__)
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_OK(x) CHECK_EQ(x, 0)

#define PEER_COUNT 4

#define SHM_NAME "/cuda-checkpoint-demo"
#define FILE_NAME "/dev/shm" SHM_NAME
#define CUDA_CHECKPOINT_JOB_FILE "CUDA_CHECKPOINT_JOB_FILE"
#define CMD_LINE "cp $" CUDA_CHECKPOINT_JOB_FILE " " FILE_NAME

void check(bool is_ok, const char *msg, const char *file, int line)
{
    if (!is_ok) {
        fprintf(stderr, "Error: \"%s\" failed at %s:%d\n", msg, file, line);
        abort();
    }
}

typedef struct shared_st
{
    pthread_barrier_t barrier;
    CUipcMemHandle handles[PEER_COUNT];
} shared_t;

shared_t *shared_create(void)
{
    shared_t *shared = mmap(NULL,
                            sizeof *shared,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED | MAP_ANONYMOUS, 
                            -1,
                            0);
    CHECK(shared != MAP_FAILED);

    // allocate interprocess barrier
    pthread_barrierattr_t attrs;
    CHECK_OK(pthread_barrierattr_init(&attrs));
    CHECK_OK(pthread_barrierattr_setpshared(&attrs, 1));

    CHECK_OK(pthread_barrier_init(&shared->barrier, &attrs, PEER_COUNT + 1));

    CHECK_OK(pthread_barrierattr_destroy(&attrs));
    return shared;
}

void peer_verify_buffers(CUdeviceptr *buffers)
{
    for (int i = 0; i < PEER_COUNT; i++) {
        int value;
        CHECK_OK(cuMemcpyDtoH(&value, buffers[i], sizeof value));
        CHECK_EQ(value, i + 1);
    }
}

int peer_proc(shared_t *shared, int idx)
{
    CUcontext ctx;
    CHECK_OK(cuInit(0));
    CHECK_OK(cuDevicePrimaryCtxRetain(&ctx, 0));
    CHECK_OK(cuCtxSetCurrent(ctx));

    CUdeviceptr buffers[PEER_COUNT];

    int value = idx + 1;
    CHECK_OK(cuMemAlloc(&buffers[idx], sizeof value));
    CHECK_OK(cuMemcpyHtoD(buffers[idx], &value, sizeof value));
    CHECK_OK(cuIpcGetMemHandle(&shared->handles[idx], buffers[idx]));

    pthread_barrier_wait(&shared->barrier); // wait for all peers to populate their mem handles

    for (int i = 0; i < PEER_COUNT; i++) {
        if (i != idx) {
            CHECK_OK(cuIpcOpenMemHandle(&buffers[i], shared->handles[i], CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
        }
    }

    peer_verify_buffers(buffers);

    pthread_barrier_wait(&shared->barrier); // sync before checkpoint
    pthread_barrier_wait(&shared->barrier); // sync after restore

    peer_verify_buffers(buffers);

    return 0;
}

void configure_env(void)
{
    // Save the file created by cuda-checkpoint --launch-job and set CUDA_CHECKPOINT_JOB_FILE in the parent environment.
    // This allows jobs to be launched without becoming children of cuda-checkpoint.
    CHECK_OK(close(shm_open(SHM_NAME, O_RDWR | O_CREAT | O_TRUNC, 0666)));
    char *argv[] = {"cuda-checkpoint", "--launch-job", "bash", "-c", CMD_LINE, NULL};
    int child = fork();
    CHECK(child >= 0);

    if (child == 0) {
        CHECK_OK(execvp(argv[0], argv));
    }

    int status = 0;
    CHECK_EQ(waitpid(child, &status, 0), child);
    CHECK(WIFEXITED(status));
    CHECK_OK(WEXITSTATUS(status));

    CHECK_OK(setenv(CUDA_CHECKPOINT_JOB_FILE, FILE_NAME, 0));
}

int main(int argc, char **argv)
{
    bool should_configure_env = false;
    if (argc == 2 && strcmp(argv[1], "--configure-env") == 0) {
        should_configure_env = true;
    }

    if (getenv(CUDA_CHECKPOINT_JOB_FILE) == NULL && !should_configure_env) {
        fprintf(stderr,
                "Either launch with cuda-checkpoint like so:\n"
                "    cuda-checkpoint --launch-job %s\n"
                "or request that this program configure the environment like so:\n"
                "    %s --configure-env\n",
                argv[0],
                argv[0]);
        exit(1);
    }

    if (should_configure_env) {
        configure_env();
    }

    pid_t pids[PEER_COUNT];
    shared_t *shared = shared_create();
    for (int i = 0; i < PEER_COUNT; i++) {
        pid_t pid = fork();
        CHECK(pid >= 0);
        if (pid == 0) {
            exit(peer_proc(shared, i));
        }
        pids[i] = pid;
    }


    pthread_barrier_wait(&shared->barrier); // peer mem handle barrier

    pthread_barrier_wait(&shared->barrier); // sync before checkpoint
    printf("Checkpointing... ");

    // Sequentially go through all the processes in the job

    // just like CRIU, do lock and checkpoint passes separately
    for (int i = 0; i < PEER_COUNT; i++) {
        CUcheckpointLockArgs lock_args = {0};
        CHECK_OK(cuCheckpointProcessLock(pids[i], &lock_args));
    }

    for (int i = 0; i < PEER_COUNT; i++) {
        CUcheckpointCheckpointArgs checkpoint_args = {0};
        CHECK_OK(cuCheckpointProcessCheckpoint(pids[i], &checkpoint_args));
    }

    // Restore every process before unlocking any of them.
    // Restore order does not need to match the order used for checkpoint.
    // After every process has been restored, unlock them all so they can
    // start using CUDA again.
    for (int i = 0; i < PEER_COUNT; i++) {
        CUcheckpointRestoreArgs restore_args = {0};
        CHECK_OK(cuCheckpointProcessRestore(pids[i], &restore_args));
    }

    for (int i = 0; i < PEER_COUNT; i++) {
        CUcheckpointUnlockArgs unlock_args = {0};
        CHECK_OK(cuCheckpointProcessUnlock(pids[i], &unlock_args));
    }

    printf("Restored!\n");
    pthread_barrier_wait(&shared->barrier); // sync after restore


    for (int i = 0; i < PEER_COUNT; i++) {
        int status = 0;
        CHECK_EQ(waitpid(pids[i], &status, 0), pids[i]);
        CHECK(WIFEXITED(status));
        CHECK_OK(WEXITSTATUS(status));
    }

    if (should_configure_env) {
        CHECK_OK(unlink(FILE_NAME));
    }

    printf("Success!\n");
    return 0;
}
