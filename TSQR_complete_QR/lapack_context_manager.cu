#include "lapack_context_manager.h"

cusolverDnHandle_t get_solver_handle()
{
  static cusolverDnHandle_t handle;
  static int initialized = 0;

  if(!initialized)
  {
    cusolverStatus_t stat;
    stat = cusolverDnCreate(&handle);

    if (stat != CUSOLVER_STATUS_SUCCESS)
    {
      printf("CUSOLVER initialization failed! Status: %d\n", stat);
      initialized = 0;
    }
    else
      initialized = 1;
  }
  return handle;
}
