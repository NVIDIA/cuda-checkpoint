/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 10000

__device__ int counter = 100;
__global__ void increment()
{
    counter++;
}

int main(void)
{
    cudaFree(0);

    int sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
    sockaddr_in addr = {AF_INET, htons(PORT), inet_addr("127.0.0.1")};
    bind(sock, (sockaddr *)&addr, sizeof addr);

    while (true) {
        char buffer[16] = {0};
        sockaddr_in peer = {0};
        socklen_t inetSize = sizeof peer;
        int hCounter = 0;

        recvfrom(sock, buffer, sizeof buffer, 0, (sockaddr *)&peer, &inetSize);

        increment<<<1,1>>>();
        cudaMemcpyFromSymbol(&hCounter, counter, sizeof counter);

        size_t bytes = sprintf(buffer, "%d\n", hCounter);
        sendto(sock, buffer, bytes, 0, (sockaddr *)&peer, inetSize);
    }
    return 0;
}
