#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>


///////////////////// Data Structures /////////////////////
# define MAX_SIZE SIZE_MAX

typedef enum {
    FP64,
    FP32,
    INT64,
    INT32,
    INT16,
    INT8,
    UINT64,
    UINT32,
    UINT16,
    UINT8
} dtype_t;


typedef struct {
    void*       data;
    dtype_t     dtype;
    size_t      size;
    size_t*     shape;
    size_t*     strides;
    size_t      rank;
} tensor;

typedef enum {
    ERR_OK = 0,
    // pointer / argument errors
    ERR_NULL_PTR,
    ERR_INVALID_ARG,
    // shape / dimension errors
    ERR_INVALID_SHAPE,
    ERR_INVALID_RANK,
    ERR_INVALID_AXIS,
    ERR_SHAPE_MISMATCH,
    ERR_NOT_BROADCASTABLE,
    // dtype errors
    ERR_INVALID_DTYPE,
    ERR_DTYPE_MISMATCH,
    // memory / system
    ERR_MALLOC_FAIL,
    // indexing
    ERR_OUT_OF_BOUNDS,
    // functionality
    ERR_NOT_IMPLEMENTED,
    // fallback
    ERR_UNKNOWN
} error_code_t;

typedef struct {
    error_code_t code;
    const char*  msg;
} error_t;

/////////////////////////////////////////////////////////////

///////////////////// Function Prototypes /////////////////////

// 1. Memory Management <mem>
tensor* tensor_mem_alloc (size_t rank, const size_t* shape, dtype_t dtype, error_t* error);
tensor* tensor_mem_init  (size_t rank, const size_t* shape, dtype_t dtype, const void* data, error_t* error);
void    tensor_mem_free  (tensor* t);
tensor* tensor_mem_copy  (const tensor* t, error_t* error);

// 2. Metadata <meta>
dtype_t tensor_meta_dtype (const tensor* t, error_t* error);
size_t  tensor_meta_size  (const tensor* t, error_t* error);
size_t* tensor_meta_shape (const tensor* t, error_t* error);
size_t  tensor_meta_rank  (const tensor* t, error_t* error);

// 3. Operations <op>
// 3.1. Elementwise Arithmetic <ew>
// 3.1.1. Primitives <prim>
typedef enum {
    ADD,
    SUB,
    MUL,
    DIV,
    REM,
    NEG,
    ABS,
    FMA,
    MIN,
    MAX,
    EQ,
    NEQ,
    GT,
    GTE,
    LT,
    LTE,
    AND,
    OR,
    XOR,
    NOT,
    SHL,
    SHR,
    SAR,
    CLZ,
    CTZ,
    POPCNT
} pri_op_t;

tensor* tensor_op_ew_prim(pri_op_t op, const tensor** inputs, error_t* error);

// 3.1.2. Kernels <ker>
typedef void (*ew_ker_t)(
    void** inputs,
    void*  output,
    size_t n
);
tensor* tensor_op_ew_ker(ew_ker_t kernel, const tensor** inputs, error_t* error);

// 3.2. Reduction <rdc>
// 3.2.1. Primitives <prim>
tensor* tensor_op_rdc_prim(pri_op_t op, const tensor* t, size_t axis, error_t* error);

// 3.2.2. Kernels <ker>
tensor* tensor_op_rdc_ker(ew_ker_t kernel, const tensor* t, size_t axis, error_t* error);

// 3.3. Views <view>
tensor* tensor_op_view_reshape(tensor* t, size_t new_rank, const size_t* new_shape, error_t* error);
tensor* tensor_op_view_permute(tensor* t, const size_t* order, error_t* error);
tensor* tensor_op_view_slice  (tensor* t, const size_t* start, const size_t* stop, const size_t* step, error_t* error);
tensor* tensor_op_view_expand (tensor* t, size_t new_rank, const size_t* new_shape, error_t* error);


// 4. Utility Functions <util>
void*  tensor_util_dptr (const tensor* t, error_t* error);
size_t tensor_util_bsize(const tensor* t, error_t* error);

// 4.1. Internal Utilities <_util_>
size_t tensor__util__dtype_size       (dtype_t dtype);
bool   tensor__util__is_contiguous    (const tensor* t, error_t* error);
size_t tensor__util__numel            (const tensor* t, error_t* error);
size_t tensor__util__offset_from_index(const tensor* t, const size_t* indices, error_t* error);
bool   tensor__util__shape_equal      (const tensor* a, const tensor* b, error_t* error);
bool   tensor__util__is_broadcastable (const tensor* a, const tensor* b, error_t* error);
void   tensor__util__compute_strides  (size_t rank, const size_t* shape, size_t* strides);

// 4.2. Sheduling/Performance Utilities <__util__>

// 5. Higher Level Operations
// 5.1. Linear Algebra <linalg>
tensor* tensor_linalg_matmul      (const tensor* a, const tensor* b, error_t* error);
tensor* tensor_linalg_dot         (const tensor* a, const tensor* b, error_t* error);
tensor* tensor_linalg_mmac        (const tensor* a, const tensor* b, const tensor* c, double alpha, double beta, error_t* error);
tensor* tensor_linalg_transpose   (const tensor* t, const size_t* perm, error_t* error);
tensor* tensor_linalg_trace       (const tensor* t, error_t* error);
tensor* tensor_linalg_diag        (const tensor* t, error_t* error);
tensor* tensor_linalg_eye         (size_t n, dtype_t dtype, error_t* error);
tensor* tensor_linalg_norm        (const tensor* t, int ord, error_t* error);
tensor* tensor_linalg_solve_linear(const tensor* a, const tensor* b, error_t* error);
tensor* tensor_linalg_inverse     (const tensor* t, error_t* error);
tensor* tensor_linalg_cholesky    (const tensor* t, error_t* error);
tensor* tensor_linalg_qr          (const tensor* t, tensor** q, tensor** r, error_t* error);
tensor* tensor_linalg_lu          (const tensor* t, tensor** p, tensor** l, tensor** u, error_t* error);
tensor* tensor_linalg_eig         (const tensor* t, tensor** eigenvectors, error_t* error);
tensor* tensor_linalg_svd         (const tensor* t, tensor** u, tensor** s, tensor** vt, error_t* error);

// 5.2. Data Manipulation <data>
tensor* tensor_from_const         (const size_t data, size_t rank, const size_t* shape, dtype_t dtype, error_t* error);
tensor* tensor_from_buffer        (void* data, size_t rank, const size_t* shape, dtype_t dtype, error_t* error);
tensor* tensor_from_buffer_copy   (const void* data, size_t rank, const size_t* shape, dtype_t dtype, error_t* error);
tensor* tensor_from_array_1d      (const void* data, size_t size, dtype_t dtype, error_t* error);
tensor* tensor_from_nested        (const void* nested_data, size_t rank, const size_t* shape, dtype_t dtype, error_t* error);
void*   tensor_to_buffer          (const tensor* t, error_t* error);
void*   tensor_to_buffer_copy     (const tensor* t, error_t* error);
void*   tensor_to_array_1d        (const tensor* t, error_t* error);
void*   tensor_to_nested          (const tensor* t, error_t* error);

// 5.3. Debugging/Visualization <debug>
void    tensor_print              (const tensor* t, error_t* error);
void    tensor_print_structure    (const tensor* t, error_t* error);


#endif // TENSOR_H
