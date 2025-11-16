
NVCC= nvcc
NVCC_FLAGS= --cuda-path /usr/local/cuda/ -gencode=arch=compute_60,code=\"sm_60\"
NVCC_FLAGS += -Wno-deprecated-gpu-targets

OPT_LEVEL= -O2

SHELL=/bin/bash
OBJ_DIR=obj
BIN_DIR=bin
OUT_DIR=out
SRC_DIR=src
TEST_DIR=bench

INCLUDE_DIRS= -I/usr/local/cuda/include -I./include/cblib
LD_FLAGS = -lcudart -L/usr/local/cuda/lib64

ALL_FLAGS = $(NVCC_FLAGS) $(INCLUDE_DIRS) $(OPT_LEVEL)

.PHONY: default clean debug
default: $(BIN_DIR)/cibyl

debug: NVCC_FLAGS += -g
debug: CFLAGS := $(filter-out $(OPT_LEVEL), $(CFLAGS))
debug: default

CBLIB_SRCS = src/cblib/cb_const.c src/cblib/cb_dbg.c src/cblib/cb_gen.c \
	     src/cblib/cb_lib.c src/cblib/cb_magical.c src/cblib/cb_normal.c
CBLIB_OBJS = $(CBLIB_SRCS:src/%.c=obj/%.o)
CBLIB_DEPS = $(CBLIB_OBJS:%.o=%.d)
-include $(CBLIB_DEPS)

.PHONY: cblib
cblib: $(CBLIB_OBJS)

CUDA_SRCS = src/gpu_perft_arch2.cu src/cblib_gpu/gpu_tables.cu \
		src/cblib_gpu/gpu_gen.cu src/cblib_gpu/gpu_lib.cu
CUDA_OBJS = $(CUDA_SRCS:src/%.cu=obj/%.cu.o)
CUDA_DEPS = $(CUDA_OBJS:%.cu.o=%.cu.d)
-include $(CUDA_DEPS)

.PHONY: cblib
cblib: $(CBLIB_OBJS)

$(BIN_DIR)/cibyl: obj/main.o $(CBLIB_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) -o $@ $^

obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(NVCC) $(ALL_FLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $< 

obj/%.cu.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(ALL_FLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $< 

clean:
	rm -rf $(OBJ_DIR)/* $(BIN_DIR)/*
