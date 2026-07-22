#!/bin/bash
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xeuo pipefail

rm -rf *.json *.log *.onnx *.raw *.trt bad/ good/ polygraphy_capability_dumps/

# 00-Get engines
export MODEL_TRAINED=${TRT_COOKBOOK_PATH}/00-Data/model/model-trained.onnx
export MODEL_TRAINED_SPARITY=${TRT_COOKBOOK_PATH}/00-Data/model/model-trained-sparsity.onnx
export MODEL_UNKNOWN=${TRT_COOKBOOK_PATH}/00-Data/model/model-unknown.onnx

polygraphy run \
    $MODEL_TRAINED \
    --trt \
    --save-engine ./model-trained.trt \
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
    --input-shapes   'x:[4,1,28,28]' \
    --save-inputs model-trained-inputs.raw \
    --save-outputs model-trained-outputs.raw \
    --silent

polygraphy run \
    $MODEL_TRAINED \
    --trt \
    --save-engine ./model-trained-FP16.trt \
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
    --input-shapes   'x:[4,1,28,28]' \
    --silent

# 01-Export information of the ONNX file
polygraphy inspect model \
    $MODEL_TRAINED \
    --model-type=onnx \
    --shape-inference \
    --show layers attrs weights \
    --list-unbounded-dds \
    --verbose \
    > result-01.log 2>&1

# 02-Export information of the TensorRT network
polygraphy inspect model \
    $MODEL_TRAINED \
    --model-type=onnx \
    --convert-to=trt \
    --shape-inference \
    --show layers attrs weights \
    --list-unbounded-dds \
    --verbose \
    > result-02.log 2>&1

# 03-Export information of input / output data
polygraphy inspect data \
    model-trained-inputs.raw \
    --all \
    --show-values \
    --histogram \
    --num-items 5 \
    --line-width 100 \
    > result-03.log 2>&1

polygraphy inspect data \
    model-trained-outputs.raw \
    --all \
    --show-values \
    --histogram \
    --num-items 5 \
    --line-width 100 \
    >> result-03.log 2>&1

# 04-Judge whether a ONNX file is supported by TensorRT natively
# Notice:
# `$MODEL_UNKNOWN` is not fully supportede by TensorRT
# So the output directory "polygraphy_capability_dumps" is crerated, which contains information of the subgraphs supported / unsupported by TensorRT
polygraphy inspect capability \
    $MODEL_TRAINED \
    > result-04-A.log 2>&1

polygraphy inspect capability \
    $MODEL_UNKNOWN \
    > result-04-B.log 2>&1

# 05-Check whether sparsity is supported by the model
polygraphy inspect sparsity \
    $MODEL_TRAINED_SPARITY \
    > result-05-A.log 2>&1

polygraphy inspect sparsity \
    $MODEL_TRAINED \
    > result-05-B.log 2>&1

echo "Finish"
