#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -v

# run the counter application
./counter &

#get the PID of counter
PID=$!

# wait for counter to bind to the UDP socket
sleep 1

#send a packet
echo hello | nc -u localhost 10000 -W 1

# confirm that counter is using the GPU
nvidia-smi --query --display=PIDS | grep $PID

# suspend CUDA
cuda-checkpoint --toggle --pid $PID

# confirm that counter is no longer using the GPU
nvidia-smi --query --display=PIDS | grep $PID

# create the directory which will hold the checkpoint image
mkdir -p demo

# checkpoint counter
criu dump --shell-job --images-dir demo --tree $PID

# confirm that counter is no longer running
ps --pid $PID

# restore counter
criu restore --shell-job --restore-detached --images-dir demo

# resume CUDA
cuda-checkpoint --toggle --pid $PID

# send another packet
echo hello | nc -u localhost 10000 -W 1

kill $PID
