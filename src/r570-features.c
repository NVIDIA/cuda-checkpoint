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
 * Checkpoint and Restore Demo
 * Requires display driver 570 or higher and CRIU 4.0 or higher
 *
 * A parent process launches a child.
 * The child initializes CUDA and NVML then the parent checkpoints and restores the child.
 * The child continues to use CUDA and NVML after it is restored.
 *
 * Build with the CUDA 12.8 toolkit as follows:
 * gcc -I /usr/local/cuda-12.8/include r570-features.c -o r570-features -lcuda -lnvidia-ml
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda.h>
#include <nvml.h>

#define CHECK(x) assert(x)
#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_OK(x) CHECK_EQ(x, 0)

void runParent(pid_t child, int sock, int childSock, const char *libDir);
void checkpointAndRestore(pid_t child, int childSock, const char *libDir);
void runChild(int sock);
void systemf(const char *fmt, ...);

static char scratch = '\0';

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: %s <directory containing CUDA plugin>\n", argv[0]);
        exit(1);
    }

    // Do some sanity checks before starting
    const char *libDir = argv[1];
    systemf("ls %s/cuda_plugin.so", libDir);
    systemf("which cuda-checkpoint");
    systemf("which criu");

    int socks[2];
    CHECK_OK(socketpair(AF_UNIX, SOCK_STREAM, 0, socks));

    pid_t child = fork();
    if (child == 0) {
        CHECK_OK(close(socks[1]));
        runChild(socks[0]);
        exit(0);
    }

    runParent(child, socks[1], socks[0], libDir);
    return 0;
}

void runParent(pid_t child, int sock, int childSock, const char *libDir)
{
    // wait for checkpoint request
    CHECK_EQ(read(sock, &scratch, 1), 1);

    // confirm with the CUDA driver that CUDA is running in the child
    CUprocessState state;
    CHECK_OK(cuCheckpointProcessGetState(child, &state));
    CHECK_EQ(state, CU_PROCESS_STATE_RUNNING);

    checkpointAndRestore(child, childSock, libDir);

    // unblock child upon restore
    CHECK_EQ(write(sock, &scratch, 1), 1);

    // wait for child to finish up
    CHECK_EQ(read(sock, &scratch, 1), 1);
}

void checkpointAndRestore(pid_t child, int childSock, const char *libDir)
{
    char imagesDir[] = "images-dir-XXXXXX";
    mkdtemp(imagesDir);

    struct stat sockInfo;
    CHECK_OK(fstat(childSock, &sockInfo));

    systemf("criu dump --shell-job --images-dir %s --libdir %s --ext-unix-sk=%ld --tree %d",
            imagesDir,
            libDir,
            sockInfo.st_ino,
            child);

    CHECK_EQ(waitpid(child, NULL, 0), child);

    systemf("criu restore --shell-job --images-dir %s --libdir %s --inherit-fd fd[%d]:socket:[%ld] --restore-detached",
            imagesDir,
            libDir,
            childSock,
            sockInfo.st_ino);
}

void runChild(int sock)
{
    CHECK_OK(cuInit(0));
    CHECK_OK(nvmlInit());

    CUdevice cuDev;
    CHECK_OK(cuDeviceGet(&cuDev, 0));

    int pci[3];
    CHECK_OK(cuDeviceGetAttribute(&pci[0], CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, cuDev));
    CHECK_OK(cuDeviceGetAttribute(&pci[1], CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, cuDev));
    CHECK_OK(cuDeviceGetAttribute(&pci[2], CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, cuDev));

    char busId[64];
    sprintf(busId, "%08X:%02X:%02X.0", pci[0], pci[1], pci[2]);

    nvmlDevice_t mlDev = 0;
    CHECK_OK(nvmlDeviceGetHandleByPciBusId(busId, &mlDev));

    nvmlMemory_t before = {0};
    CHECK_OK(nvmlDeviceGetMemoryInfo(mlDev, &before));

    CUcontext ctx;
    CUdeviceptr ptr;
    CHECK_OK(cuDevicePrimaryCtxRetain(&ctx, cuDev));
    CHECK_OK(cuCtxSetCurrent(ctx));

    const int magic = 0x12345678;
    CHECK_OK(cuMemAlloc(&ptr, sizeof magic));
    CHECK_OK(cuMemcpyHtoD(ptr, &magic, sizeof magic));

    // request checkpoint and wait for restore
    CHECK_EQ(write(sock, &scratch, 1), 1);
    CHECK_EQ(read(sock, &scratch, 1), 1);

    int value = 0;
    CHECK_OK(cuMemcpyDtoH(&value, ptr, sizeof value));
    CHECK_EQ(value, magic);

    nvmlMemory_t after = {0};
    CHECK_OK(nvmlDeviceGetMemoryInfo(mlDev, &after));
    CHECK(before.free > after.free);

    // unblock parent
    printf("SUCCESS\n");
    CHECK_EQ(write(sock, &scratch, 1), 1);
}

void systemf(const char *fmt, ...)
{
    char cmd[256];
    va_list lst;
    va_start(lst, fmt);
    vsprintf(cmd, fmt, lst);
    va_end(lst);
    printf("> %s\n", cmd);
    int status = system(cmd);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
