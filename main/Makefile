CC = gcc
DEBUG ?= 1
CFLAGS = -Wall -Wextra -O3 -I../include -fopenmp
ifeq ($(DEBUG),1)
    CFLAGS += -DENABLE_GRAPH_TRACE
endif

LDFLAGS = -fopenmp -lm


# HAS_AVX2 := $(shell grep -q "avx2" /proc/cpuinfo && echo yes || echo no)
HAS_AVX2 = no

ifeq ($(HAS_AVX2),yes)
    $(info >>> AVX2 detected: Using x86_omp optimized impl.)
    IMPL_KER = ../src/kers/x86_omp/prim_kers_avx2_omp.c
    IMPL_MEM = ../src/memory/x86_omp/memory_cpu_x86_omp.c
    CFLAGS += -mavx2 -mfma
else
    $(info >>> Using scalar base impl.)
    IMPL_KER = ../src/kers/sc_base/prim_kers_sc_base.c
    IMPL_MEM = ../src/memory/sc_base/memory_cpu_sc_base.c
endif


LIB_SRCS = ../src/tensor.c \
           $(IMPL_MEM) \
           $(IMPL_KER)

# LIB_SRCS = ../src/tensor.c \
#            ../src/graph/graph.c \
#            ../src/dev/graph_tracer.c \
#            ../src/kers/common/constraints.c \
#            $(IMPL_MEM) \
#            $(IMPL_KER)

LIB_OBJS = $(LIB_SRCS:.c=.o)

TARGET_EAGER = tensor_eager
TARGET_TRACE = graph_trace_test

all: $(TARGET_EAGER) $(TARGET_TRACE)



$(TARGET_EAGER): $(LIB_OBJS) main.o
	$(CC) $(LIB_OBJS) main.o -o $(TARGET_EAGER) $(LDFLAGS)



$(TARGET_TRACE): $(LIB_OBJS) dev/graph_trace_ex.o
	$(CC) $(LIB_OBJS) dev/graph_trace_ex.o -o $(TARGET_TRACE) $(LDFLAGS)


%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET_EAGER) $(TARGET_TRACE)
	find ../src -name "*.o" -delete
	find . -name "*.o" -delete
	rm -f *.dot *.png

run_eager: $(TARGET_EAGER)
	./$(TARGET_EAGER)

run_trace: $(TARGET_TRACE)
	./$(TARGET_TRACE)

.PHONY: all clean run_eager run_trace