#include <stdio.h>
#include <stdlib.h>
#include "include/tensor.h"
// #include "lazy_tensor.h"

int main() {
    printf("Starting eager tensor test...");

    extent shape[] = {4, 4};  // 4*4 matrix
    extent rank = 2;

    // 1. allocate
    tensor* tA      = tensor_alloc_matted(rank, shape, REAL32, MEM_HOST);
    tensor* tB      = tensor_alloc_matted(rank, shape, REAL32, MEM_HOST);
    tensor* tOut    = tensor_alloc_matted(rank, shape, REAL32, MEM_HOST);

    real32 val_a = 10.5f;
    real32 data_b[16] = {
        0.12f, 0.85f, 0.33f, 0.47f, 
        0.91f, 0.05f, 0.62f, 0.28f, 
        0.74f, 0.19f, 0.55f, 0.88f, 
        0.02f, 0.39f, 0.99f, 0.11f
    };
    
    // 2. fill / view from
    fill_const_ker(tA, NULL, 0, &val_a, sizeof(float));
    tensor_view_from(tB, (dptr*)data_b, MEM_HOST);

    // 3. add
    const tensor* inputs[] = {tA, tB};
    add_ker(tOut, inputs, 2, NULL, 0);

    // 4. view_to and print result
    real32* out_ptr = (real32*)tensor_view_to(tOut, MEM_HOST);
    if (out_ptr) {
        printf("\nResult Matrix (tA + tB):\n");
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                printf("%8.2f ", out_ptr[i * shape[1] + j]);
            }
            printf("\n");
        }
    }

    // 5. clean
    tensor_free(tA);
    tensor_free(tB);
    tensor_free(tOut);
    return 0;
}