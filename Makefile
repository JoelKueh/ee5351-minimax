
NVCC= nvcc
NVCC_FLAGS= -gencode=arch=compute_75,code=\"sm_75\"

OPT_LEVEL= -O2
NVCC_FLAGS+=$(OPT_LEVEL)

SHELL=/bin/bash
OBJ_DIR=obj
BIN_DIR=bin
OUT_DIR=out
SRC_DIR=src
TEST_DIR=bench

INCLUDE_DIRS= -I/usr/local/cuda/include -I./include/cblib
LD_FLAGS = -lcudart -L/usr/local/cuda/lib64

NVCC_FLAGS += $(INCLUDE_DIRS)

.PHONY: default clean debug
default: $(BIN_DIR)/cibyl

debug: NVCC_FLAGS += -g
debug: CFLAGS := $(filter-out $(OPT_LEVEL), $(CFLAGS))
debug: default

### cblib compilation

CBLIB_SRCS = src/cblib/cb_const.c src/cblib/cb_dbg.c src/cblib/cb_gen.c \
	     src/cblib/cb_lib.c src/cblib/cb_magical.c src/cblib/cb_normal.c
CBLIB_OBJS = $(CBLIB_SRCS:src/%.c=obj/%.o)
CBLIB_DEPS = $(CBLIB_OBJS:%.o=%.d)
-include $(CBLIB_DEPS)

.PHONY: cblib
cblib: $(CBLIB_OBJS)

### cuda kernel compilation

CUDA_SRCS = 
CUDA_OBJS = $(CUDA_SRCS:src/%.c=obj/%.o)
CUDA_DEPS = $(CUDA_OBJS:%.o=%.d)
-include $(CUDA_DEPS)

.PHONY: cblib
cblib: $(CBLIB_OBJS)

### main program compilation

$(BIN_DIR)/cibyl: src/main.o $(CBLIB_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) -o $@ $^

### utility targets

obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $< 

clean:
	rm -rf $(OBJ_DIR)/* $(BIN_DIR)/*
