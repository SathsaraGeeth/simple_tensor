#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>
#include <omp.h>
#include "include/tensor.h"

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------------- SCALAR KERNEL ----------------
void add_float_kernel_scalar(void** inputs, void* output, size_t n) {
    float* a = (float*)inputs[0];
    float* b = (float*)inputs[1];
    float* out = (float*)output;
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

// ---------------- OPTIMIZED SIMD + OPENMP KERNEL ----------------
void add_float_kernel_optimized(void** inputs, void* output, size_t n) {
    float* a = (float*)inputs[0];
    float* b = (float*)inputs[1];
    float* out = (float*)output;

    size_t simd_width = 8;          // AVX2: 8 floats per register
    size_t i;
    size_t simd_end = n - (n % simd_width);

    // #pragma omp parallel for schedule(static)
    for (i = 0; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(&a[i]);   // aligned load
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&out[i], vr);       // aligned store
    }

    // remainder
    for (; i < n; i++) out[i] = a[i] + b[i];
}

int main() {
    error_t err;
    size_t shape[2] = {1UL << 14, 1UL << 14};
    const tensor* inputs[2];

    double t0, t_init, t_alloc, t_kernel_scalar, t_kernel_optimized, t_to_array, t_copy;

    // ---------------- INIT ----------------
    t0 = now_sec();
    tensor* t1 = tensor_mem_init_const(2, shape, REAL32, 1, &err);
    tensor* t2 = tensor_mem_init_const(2, shape, REAL32, 10, &err);
    t_init = now_sec() - t0;

    if (!t1 || !t2) { printf("Init error: %s\n", err.msg); return 1; }
    inputs[0] = t1; inputs[1] = t2;

    // copy
    t0 = now_sec();
    tensor* t7 = tensor_mem_copy(t1, &err);
    t_copy = now_sec() - t0;
    

    // ---------------- ALLOC ----------------
    t0 = now_sec();
    tensor* t3 = tensor_mem_alloc(2, shape, REAL32, &err);
    t_alloc = now_sec() - t0;

    if (!t3) { printf("Alloc error: %s\n", err.msg); goto cleanup; }

    // ---------------- KERNEL SCALAR ----------------
    t0 = now_sec();
    tensor_op_ew_ker(add_float_kernel_scalar, t3, inputs, 2, &err);
    t_kernel_scalar = now_sec() - t0;

    // ---------------- KERNEL OPTIMIZED ----------------
    t0 = now_sec();
    tensor_op_ew_ker(add_float_kernel_optimized, t3, inputs, 2, &err);
    t_kernel_optimized = now_sec() - t0;

    // ---------------- TO ARRAY ----------------
    t0 = now_sec();
    float* out_data = (float*)tensor_mem_view_to_array(t3, &err);
    t_to_array = now_sec() - t0;

    if (!out_data) { printf("Array error: %s\n", err.msg); goto cleanup; }

    // ---------------- TIMINGS ----------------
    printf("\nTimings (seconds):\n");
    printf("init:              %f\n", t_init);
    printf("alloc:             %f\n", t_alloc);
    printf("kernel scalar:     %f\n", t_kernel_scalar);
    printf("kernel optimized:  %f\n", t_kernel_optimized);
    printf("to_array:          %f\n", t_to_array);
    printf("copy:              %f\n", t_copy);

cleanup:
    free(out_data);
    tensor_mem_free(t1);
    tensor_mem_free(t2);
    tensor_mem_free(t3);

    return 0;
}