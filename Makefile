
NVCC= nvcc
NVCC_FLAGS= -gencode=arch=compute_75,code=\"sm_75\" -Wno-deprecated-gpu-targets

C_FLAGS=

OPT_LEVEL= -O2

SHELL=/bin/bash
OBJ_DIR=obj
BIN_DIR=bin
OUT_DIR=out
SRC_DIR=src
TEST_DIR=bench

INCLUDE_DIRS= -I/usr/local/cuda/include -I./include/cblib -I./include/utils
LD_FLAGS = -lcudart -L/usr/local/cuda/lib64

.PHONY: default clean debug
default: $(BIN_DIR)/debug

debug: OPT_LEVEL= 
debug: NVCC_FLAGS += -g -G
debug: CFLAGS += -g
debug: default

ALL_FLAGS = $(NVCC_FLAGS) $(INCLUDE_DIRS) $(OPT_LEVEL)
CFLAGS = $(INCLUDE_DIRS) $(OPT_LEVEL)

CBLIB_SRCS = src/cblib/cb_const.c src/cblib/cb_dbg.c src/cblib/cb_gen.c \
	     src/cblib/cb_lib.c src/cblib/cb_magical.c src/cblib/cb_normal.c \
		 src/perft_cpu.c
CBLIB_OBJS = $(CBLIB_SRCS:src/%.c=obj/%.o)
CBLIB_DEPS = $(CBLIB_OBJS:%.o=%.d)
-include $(CBLIB_DEPS)

.PHONY: cblib
cblib: $(CBLIB_OBJS)

CUDA_SRCS = src/perft_gpu.cu
CUDA_OBJS = $(CUDA_SRCS:src/%.cu=obj/%.cu.o)
CUDA_DEPS = $(CUDA_OBJS:%.cu.o=%.cu.d)
-include $(CUDA_DEPS)

.PHONY: cblib
cblib: $(CBLIB_OBJS)

$(BIN_DIR)/debug: obj/debug.o $(CBLIB_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(ALL_FLAGS) -o $@ $^

obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $< 

obj/%.cu.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(ALL_FLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $< 

clean:
	rm -rf ./$(OBJ_DIR)/* ./$(BIN_DIR)/*
