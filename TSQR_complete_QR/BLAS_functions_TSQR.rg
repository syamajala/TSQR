import "regent"

local clib = regentlib.c
local cmath = terralib.includec("math.h")
local std = terralib.includec("stdlib.h")
local cstr = terralib.includec("string.h")

--BLAS info
local blas_lib = terralib.includecstring [[

  extern void dgeqrf_(int* M, int* N, double* A, int* lda, double* tau, double* work, int* lwork, int* info);

  extern void dorgqr_(int* M, int* N, int* K, double* A, int* lda, double* tau, double* work, 
                      int* lwork, int* info);

]]

-- terralib.linklibrary("libblas.so")
-- terralib.linklibrary("liblapack.so")
assert(os.getenv("BLAS_INCLUDE_DIRS") ~= nil, "Missing BLAS header.")
assert(os.getenv("BLAS_LIB") ~= nil, "Missing BLAS library.")
terralib.linklibrary(os.getenv("BLAS_LIB"))
local cblas = terralib.includec("cblas.h", {"-I", os.getenv("BLAS_INCLUDE_DIRS")})

assert(os.getenv("LAPACK_INCLUDE_DIRS") ~= nil, "Missing LAPACK header.")
assert(os.getenv("LAPACK_LIB") ~= nil, "Missing LAPACK library.")
terralib.linklibrary(os.getenv("LAPACK_LIB"))
local lapacke = terralib.includec("lapacke.h", {"-I", os.getenv("LAPACK_INCLUDE_DIRS")})

local blas_exp = {}

--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
--~~      BLAS/LAPACK functions              ~~~~
--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
terra get_executing_processor(runtime : clib.legion_runtime_t)
  var ctx = clib.legion_runtime_get_context()
  var result = clib.legion_runtime_get_executing_processor(runtime, ctx)
  clib.legion_context_destroy(ctx)
  return result
end
get_executing_processor.replicable = true

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
end

raw_ptr = raw_ptr_factory(double)
raw_ptr_int = raw_ptr_factory(int)

terra get_raw_ptr_1d(rect : rect1d,
                     pr   : clib.legion_physical_region_t,
                     fld  : clib.legion_field_id_t)
  var fa = clib.legion_physical_region_get_field_accessor_array_1d(pr, fld)
  var subrect : clib.legion_rect_1d_t
  var offsets : clib.legion_byte_offset_t[1]
  var ptr = clib.legion_accessor_array_1d_raw_rect_ptr(fa, rect, &subrect, offsets)
  clib.legion_accessor_array_1d_destroy(fa)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[0].offset / sizeof(double) }
end

terra get_raw_ptr_2d(rect : rect2d,
                     pr   : clib.legion_physical_region_t,
                     fld  : clib.legion_field_id_t)
  var fa = clib.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var subrect : clib.legion_rect_2d_t
  var offsets : clib.legion_byte_offset_t[2]
  var ptr = clib.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  clib.legion_accessor_array_2d_destroy(fa)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

terra dgeqrf_cpu_terra(
    layout : int,
    M      : int,
    N      : int,
	rectA  : rect2d,
    prA    : clib.legion_physical_region_t,
	fldA   : clib.legion_field_id_t,
    TAU    : &double)
  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  return lapacke.LAPACKE_dgeqrf(layout, M, N, rawA.ptr, rawA.offset, TAU)
end

terra dorgqr_cpu_terra(
    layout : int,
    M      : int,
    N      : int,
    K      : int,
	rectA  : rect2d,
    prA    : clib.legion_physical_region_t,
	fldA   : clib.legion_field_id_t,
    TAU    : &double)
  var rawA = get_raw_ptr_2d(rectA, prA, fldA)
  return lapacke.LAPACKE_dorgqr(layout, M, N, K, rawA.ptr, rawA.offset, TAU)
end

terra alloc_double_cpu(size : int)
  return [&double](clib.malloc(sizeof(double)*size))
end

use_cuda = false
use_hip = false
for k, v in pairs(arg) do
  if v == "-fgpu" then
    if arg[k+1] == "cuda" then
      use_cuda = true
    elseif arg[k+1] == "hip" then
      use_hip = true
    end
  end
end

if use_cuda then
  local cuda_home = os.getenv("CUDA_HOME")
  terralib.includepath = terralib.includepath .. ";" .. cuda_home .. "/include"

  terralib.linklibrary(cuda_home .. "/lib64/libcusolver.so")
  terralib.linklibrary("./libcontext.so")

  cuda_runtime = terralib.includec("cuda_runtime.h")
  local cusolver = terralib.includec("cusolverDn.h")

  local mgr = terralib.includec("lapack_context_manager.h", {"-I", "./"})

  terra alloc_double_gpu(size : int)
    var IPIV_temp : &double
    cuda_runtime.cudaMalloc([&&opaque](&IPIV_temp), sizeof(double)*size)
    return IPIV_temp
  end

  terra dgeqrf_gpu_terra(
      layout : int,
      M      : int,
      N      : int,
      rectA  : rect2d,
      prA    : clib.legion_physical_region_t,
      fldA   : clib.legion_field_id_t,
      TAU    : &double)

    var handle : cusolver.cusolverDnHandle_t = mgr.get_solver_handle()

    var stream : cuda_runtime.cudaStream_t
    cuda_runtime.cudaStreamCreate(&stream)
    cusolver.cusolverDnSetStream(handle, stream)

    var rawA = get_raw_ptr_2d(rectA, prA, fldA)

    var lwork : int
    cusolver.cusolverDnDgeqrf_bufferSize(handle, M, N, rawA.ptr, rawA.offset, &lwork)

    var d_work : &double
    cuda_runtime.cudaMalloc([&&opaque](&d_work), sizeof(double) * lwork)

    var d_info : &int
    cuda_runtime.cudaMalloc([&&opaque](&d_info), sizeof(int))

    var ret = cusolver.cusolverDnDgeqrf(handle, M, N, rawA.ptr, rawA.offset, TAU, d_work, lwork, d_info)
    cuda_runtime.cudaFree(d_work)
    cuda_runtime.cudaFree(d_info)
    cuda_runtime.cudaStreamDestroy(stream)
    return ret
  end

  terra dorgqr_gpu_terra(
      layout : int,
      M      : int,
      N      : int,
      K      : int,
      rectA  : rect2d,
      prA    : clib.legion_physical_region_t,
      fldA   : clib.legion_field_id_t,
      TAU    : &double)

    var handle : cusolver.cusolverDnHandle_t = mgr.get_solver_handle()

    var stream : cuda_runtime.cudaStream_t
    cuda_runtime.cudaStreamCreate(&stream)
    cusolver.cusolverDnSetStream(handle, stream)

    var rawA = get_raw_ptr_2d(rectA, prA, fldA)

    var lwork : int
    cusolver.cusolverDnDorgqr_bufferSize(handle, M, N, K, rawA.ptr, rawA.offset, TAU, &lwork)

    var d_work : &double
    cuda_runtime.cudaMalloc([&&opaque](&d_work), sizeof(double) * lwork)

    var d_info : &int
    cuda_runtime.cudaMalloc([&&opaque](&d_info), sizeof(int))

    var ret = cusolver.cusolverDnDorgqr(handle, M, N, K, rawA.ptr, rawA.offset, TAU, d_work, lwork, d_info)
    cuda_runtime.cudaFree(d_work)
    cuda_runtime.cudaFree(d_info)
    cuda_runtime.cudaStreamDestroy(stream)
    return ret
  end
elseif use_hip then
  local hip_home = os.getenv("HIP_PATH") .. "/../"
  terralib.includepath = terralib.includepath .. ";" .. hip_home .. "/rocblas/include"
  terralib.includepath = terralib.includepath .. ";" .. hip_home .. "/rocsolver/include"

  terralib.linklibrary(hip_home .. "/rocsolver/lib/librocsolver.so")
  terralib.linklibrary("./libcontext.so")

  hip_runtime = terralib.includecstring [[
    #define __HIP_PLATFORM_HCC__ 1
    #include <hip/hip_runtime.h>
    hipStream_t hipGetTaskStream();
]]
  local rocblas = terralib.includec("rocblas.h")
  local rocsolver = terralib.includec("rocsolver.h")

  local mgr = terralib.includec("blas_context_manager.h", {"-I", "./", "-D__HIPCC__"})

  terra alloc_double_gpu(size : int)
    var IPIV_temp : &double
    hip_runtime.hipMalloc([&&opaque](&IPIV_temp), sizeof(double)*size)
    return IPIV_temp
  end

  terra dgeqrf_gpu_terra(
      layout : int,
      M      : int,
      N      : int,
      rectA  : rect2d,
      prA    : clib.legion_physical_region_t,
      fldA   : clib.legion_field_id_t,
      TAU    : &double)

    var handle : rocblas.rocblas_handle = mgr.get_blas_handle()

    var stream : hip_runtime.hipStream_t = hip_runtime.hipGetTaskStream()
    rocblas.rocblas_set_stream(handle, stream)

    var rawA = get_raw_ptr_2d(rectA, prA, fldA)

    -- var d_info : &int
    -- hip_runtime.hipMalloc([&&opaque](&d_info), sizeof(int))

    var ret = rocsolver.rocsolver_dgeqrf(handle, M, N, rawA.ptr, rawA.offset, TAU)
    -- hip_runtime.hipFree(d_info)
    return ret
  end

  terra dorgqr_gpu_terra(
      layout : int,
      M      : int,
      N      : int,
      K      : int,
      rectA  : rect2d,
      prA    : clib.legion_physical_region_t,
      fldA   : clib.legion_field_id_t,
      TAU    : &double)

    var handle : rocblas.rocblas_handle = mgr.get_blas_handle()

    var stream : hip_runtime.hipStream_t = hip_runtime.hipGetTaskStream()
    rocblas.rocblas_set_stream(handle, stream)

    var rawA = get_raw_ptr_2d(rectA, prA, fldA)

    -- var d_info : &int
    -- hip_runtime.hipMalloc([&&opaque](&d_info), sizeof(int))

    var ret = rocsolver.rocsolver_dorgqr(handle, M, N, K, rawA.ptr, rawA.offset, TAU)
    -- hip_runtime.hipFree(d_info)
    return ret
  end
end

--task to copy the upper diagonal results of the dgeqrf BLAS subroutine to corresponding R region
__demand(__cuda, __local)
task get_R_matrix(p        : int,
                  blocks   : int,
                  n        : int,
                  matrix   : region(ispace(int2d), double),
                  R_matrix : region(ispace(int2d), double))
where reads(matrix), writes(R_matrix)
do
  var r_point : int2d
  var x_shift : int
  var offset : int = 0

  for j = 0, p do
    offset += blocks
  end

  for i in matrix do
    x_shift = i.x - offset
    if i.y >= x_shift then
      r_point = {x = i.x - offset + p*n, y = i.y}
      R_matrix[r_point] = matrix[i]
    end
  end
end

__demand(__cuda)
task blas_exp.local_qr_factorize(layout : int,
                                 p      : int,
                                 blocks : int,
                                 m      : int,
                                 n      : int,
                                 A      : region(ispace(int2d), double),
                                 R      : region(ispace(int2d), double))
where
  reads(A), writes(R)
do
  var rectA = A.bounds
  var sizeA = rectA.hi - rectA.lo + {1, 1}
  var M = sizeA.x
  var N = sizeA.y
  var K = 0
  if M > N then
    K = N
  else
    K = M
  end

  var proc = get_executing_processor(__runtime())

  if clib.legion_processor_kind(proc) == clib.TOC_PROC then
    [(function ()
        if use_cuda then
            return rquote
                regentlib.assert(layout == lapacke.LAPACK_COL_MAJOR, 'Expected column major layout.')
                var TAU = alloc_double_gpu(K)
                dgeqrf_gpu_terra(layout, M, N, rectA, __physical(A)[0], __fields(A)[0], TAU)
                get_R_matrix(p, blocks, n, A, R)
                dorgqr_gpu_terra(layout, M, N, K, rectA, __physical(A)[0], __fields(A)[0], TAU)
                cuda_runtime.cudaFree(TAU)
                   end
          elseif use_hip then
            return rquote
                regentlib.assert(layout == lapacke.LAPACK_COL_MAJOR, 'Expected column major layout.')
                var TAU = alloc_double_gpu(K)
                dgeqrf_gpu_terra(layout, M, N, rectA, __physical(A)[0], __fields(A)[0], TAU)
                get_R_matrix(p, blocks, n, A, R)
                dorgqr_gpu_terra(layout, M, N, K, rectA, __physical(A)[0], __fields(A)[0], TAU)
                hip_runtime.hipFree(TAU)
                   end
          else
            return rquote regentlib.assert(false, "Build with CUDA support.") end
          end
     end)()]
  else
    var TAU = alloc_double_cpu(K)
    dgeqrf_cpu_terra(layout, M, N, rectA, __physical(A)[0], __fields(A)[0], TAU)
    get_R_matrix(p, blocks, n, A, R)
    dorgqr_cpu_terra(layout, M, N, K, rectA, __physical(A)[0], __fields(A)[0], TAU)
    clib.free(TAU)
  end
end

return blas_exp
