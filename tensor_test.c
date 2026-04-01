#include <stdio.h>
#include "include/tensor.h"

void add_float_kernel(void** inputs, void* output) {
    const float a = *(float*)inputs[0];
    const float b = *(float*)inputs[1];
    *(float*)output = a + b;
}

int main() {
    error_t err;
    size_t shape[2] = {2, 3};
    tensor* t1 = tensor_mem_init_const(2, shape, REAL32, 1, &err);
    tensor* t2 = tensor_mem_init_const(2, shape, REAL32, 10, &err);
    if (!t1 || !t2) { printf("Init error: %s\n", err.msg); return 1; }

    tensor* t3 = tensor_mem_alloc(2, shape, REAL32, &err);
    if (!t3) { printf("Alloc error: %s\n", err.msg); tensor_mem_free(t1); tensor_mem_free(t2); return 1; }

    const tensor* inputs[2] = {t1, t2};
    tensor* result = tensor_op_ew_ker(add_float_kernel, t3, inputs, 2, &err);
    if (!result) { printf("Kernel error: %s\n", err.msg); tensor_mem_free(t1); tensor_mem_free(t2); tensor_mem_free(t3); return 1; }

    float* out_data = (float*)tensor_mem_to_array(result, &err);
    if (!out_data) { printf("Array error: %s\n", err.msg); tensor_mem_free(t1); tensor_mem_free(t2); tensor_mem_free(t3); return 1; }

    for (size_t i = 0; i < result->size; i++) printf("%f ", out_data[i]);
    printf("\n");

    free(out_data);
    tensor_mem_free(t1);
    tensor_mem_free(t2);
    tensor_mem_free(t3);
    return 0;
}