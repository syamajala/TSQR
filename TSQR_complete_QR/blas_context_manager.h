// Copyright 2019 Stanford University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __CONTEXT_MANAGER_H__
#define __CONTEXT_MANAGER_H__

#include <stdio.h>
#ifdef __NVCC__
#include <cuda_runtime.h>
#include "cublas_v2.h"
#elif __HIPCC__
#include "rocblas.h"
#include "rocsolver.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __NVCC__
cublasHandle_t get_blas_handle();
#elif __HIPCC__
rocblas_handle get_blas_handle();
int job_to_svect(char *j);
#endif

#ifdef __cplusplus
}
#endif

#endif
