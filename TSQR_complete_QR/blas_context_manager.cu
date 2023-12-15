#include "blas_context_manager.h"

#ifdef __NVCC__
cublasHandle_t get_blas_handle()
{
  static cublasHandle_t handle;
  static int initialized = 0;

  if(!initialized)
  {
    cublasStatus_t stat;
    stat = cublasCreate(&handle);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      printf("CUBLAS initialization failed! Status: %d\n", stat);
      initialized = 0;
    }
    else
      initialized = 1;
  }
  return handle;
}
#elif __HIPCC__
rocblas_handle get_blas_handle()
{
  static rocblas_handle handle;
  static int initialized = 0;

  if(!initialized)
  {
    rocblas_status stat;
    stat = rocblas_create_handle(&handle);

    if (stat != rocblas_status_success)
    {
      printf("ROCBLAS initialization failed! Status: %d\n", stat);
      initialized = 0;
    }
    else
      initialized = 1;
  }
  return handle;
}

int job_to_svect(char *j)
{
  if (strcmp(j, "A") == 0)
  {
    return rocblas_svect_all;
  }
  else if(strcmp(j, "S") == 0)
  {
    return rocblas_svect_singular;
  }
  else if(strcmp(j, "O") == 0)
  {
    return rocblas_svect_overwrite;
  }
  else if(strcmp(j, "N") == 0)
  {
    return rocblas_svect_none;
  }
}
#endif
