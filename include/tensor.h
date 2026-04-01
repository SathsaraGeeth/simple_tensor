#ifndef TENSOR_H
#define TENSOR_H

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


///////////////////// Data Structures /////////////////////
# define MAX_SIZE SIZE_MAX

typedef enum {
    REAL64,
    REAL32,
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
    size_t      size;       // numel
    dtype_t     dtype;
    size_t      dtype_size;
    size_t*     shape;
    size_t*     strides;
    size_t      rank;
    bool        owns_data;
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
tensor* tensor_mem_alloc      (size_t rank, const size_t* shape, dtype_t dtype, error_t* error);
tensor* tensor_mem_init       (size_t rank, const size_t* shape, dtype_t dtype, const void* data, error_t* error);
tensor* tensor_mem_view_init  (size_t rank, const size_t* shape, dtype_t dtype, const void* data, error_t* error); // <view>
tensor* tensor_mem_init_const (size_t rank, const size_t* shape, dtype_t dtype, const size_t const_data, error_t* error);
void    tensor_mem_free       (tensor* t);
tensor* tensor_mem_copy       (const tensor* t, error_t* error);
void*   tensor_mem_to_array   (const tensor* t, error_t* error);
void*   tensor_mem_view_to_array   (const tensor* t, error_t* error);

// 2. Metadata <meta>
dtype_t tensor_meta_dtype (const tensor* t, error_t* error);
size_t  tensor_meta_size  (const tensor* t, error_t* error);
size_t* tensor_meta_shape (const tensor* t, error_t* error);
size_t  tensor_meta_rank  (const tensor* t, error_t* error);


// 3. Operations <op>
// 3.1. Elementwise Arithmetic <ew>
typedef void (*ew_ker_t)(
    void** inputs,
    void*  output,
    size_t n
);
tensor* tensor_op_ew_ker(ew_ker_t kernel, tensor* output, const tensor** inputs, error_t* error);

// 3.2. Reduction <rdc>
tensor* tensor_op_rdc_ker(ew_ker_t kernel, const tensor* t, size_t axis, error_t* error);

// 3.3. Views <view>
tensor* tensor_op_view_reshape(tensor* t, size_t new_rank, const size_t* new_shape, error_t* error);
tensor* tensor_op_view_permute(tensor* t, const size_t* order, error_t* error);
tensor* tensor_op_view_slice  (tensor* t, const size_t* start, const size_t* stop, const size_t* step, error_t* error);
tensor* tensor_op_view_expand (tensor* t, size_t new_rank, const size_t* new_shape, error_t* error);


// 4. Utility Functions <util>
// 4.1. Internal Utilities <_util_>
const  void*  tensor__util__dptr      (const tensor* t, error_t* error);
size_t tensor__util__dtype_size       (dtype_t dtype, error_t* error);
bool   tensor__util__is_contiguous    (const tensor* t, error_t* error);
size_t tensor__util__offset_from_index(const tensor* t, const size_t* indices, error_t* error);
bool   tensor__util__shape_equal      (const tensor* a, const tensor* b, error_t* error);
bool   tensor__util__is_broadcastable (const tensor* a, const tensor* b, error_t* error);
void   tensor__util__compute_strides  (size_t rank, const size_t* shape, size_t* strides, error_t* error);


#endif // TENSOR_H
