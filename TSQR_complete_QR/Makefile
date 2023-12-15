GEN_GPU_SRC	:=
LD_FLAGS :=
OUTDIR := ./
CONTEXT_LIB := libcontext.so
NVCC := nvcc
HIPCC := hipcc

ifeq ($(strip $(USE_CUDA)), 1)
GEN_GPU_SRC += blas_context_manager.cu \
				lapack_context_manager.cu
LD_FLAGS += -lcusolver -lcublas -lopenblas -L$(dir $(BLAS_LIB)) -llapacke -L$(dir $(LAPACK_LIB))
endif
ifeq ($(strip $(USE_HIP)), 1)
GEN_GPU_SRC += blas_context_manager.cu
LD_FLAGS += -lrocsolver -lrocblas -lopenblas -L/opt/rocm-5.1.0/rocblas/lib -L/opt/rocm-5.1.0/rocsolver/lib -L/sw/crusher/spack-envs/base/opt/linux-sles15-x86_64/gcc-7.5.0/openblas-0.3.17-54x7v5e4i6yxqs6j5nebrbztpy4lftj4/lib
endif

GEN_GPU_OBJS	  := $(patsubst %.cu,$(OUTDIR)/%.o,$(GEN_GPU_SRC))

ifeq ($(strip $(USE_CUDA)), 1)
$(GEN_GPU_OBJS) : $(OUTDIR)/%.o : %.cu
	$(NVCC) -Xcompiler -fPIC -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)
else ifeq ($(strip $(USE_HIP)), 1)
$(GEN_GPU_OBJS) : $(OUTDIR)/%.o : %.cu
	$(HIPCC) -fPIC -o $@ -c $< $(INC_FLAGS) $(HIPCC_FLAGS)
endif

$(CONTEXT_LIB): $(GEN_CPU_OBJS) $(GEN_GPU_OBJS)
	$(CXX) -shared $^ -o $@

all: $(CONTEXT_LIB)
