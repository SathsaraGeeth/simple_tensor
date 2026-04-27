CC = gcc
CFLAGS = -Wall -Wextra -O3 -I./include -fopenmp
LDFLAGS = -fopenmp -lm

# HAS_AVX2 := $(shell grep -q "avx2" /proc/cpuinfo && echo yes || echo no)
HAS_AVX2 = no

ifeq ($(HAS_AVX2),yes)
    $(info >>> AVX2 detected: Using x86_omp optimized impl.)
    IMPL_KER = src/kers/x86_omp/prim_kers_avx2_omp.c
    IMPL_MEM = src/memory/x86_omp/memory_cpu_x86_omp.c
    CFLAGS += -mavx2 -mfma
else
    $(info >>> Using scalar base impl.)
    IMPL_KER = src/kers/sc_base/prim_kers_sc_base.c
    IMPL_MEM = src/memory/sc_base/memory_cpu_sc_base.c
endif

SRCS = src/tensor.c $(IMPL_MEM) $(IMPL_KER) main.c
OBJS = $(SRCS:.c=.o)
TARGET = tensor_test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET)
	find src -name "*.o" -delete
	rm -f main.o

run: all
	./$(TARGET)

.PHONY: all clean run