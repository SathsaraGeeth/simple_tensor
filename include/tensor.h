#ifndef TENSOR_H
#define TENSOR_H

#include "dtype.h"
#include "memory.h"
#include <stddef.h>


typedef enum {
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    REAL32,
    REAL64
} dtype_t;

static inline extent dtype_size(dtype_t type) {
    switch (type) {
        case INT8:   return sizeof(int8);
        case INT16:  return sizeof(int16);
        case INT32:  return sizeof(int32);
        case INT64:  return sizeof(int64);
        case UINT8:  return sizeof(uint8);
        case UINT16: return sizeof(uint16);
        case UINT32: return sizeof(uint32);
        case UINT64: return sizeof(uint64);
        case REAL32: return sizeof(real32);
        case REAL64: return sizeof(real64);
        default:     return 0;
    }
}

typedef struct {
    mem_block*   data;
    mem_block*   shape;
    mem_block*   strides;
    extent       rank;
    dtype_t      dtype;
    extent       size;
    boolean      is_contiguous;
    boolean      error;
} tensor;

extern  tensor ERROR_TENSOR;


typedef tensor* (*ker_t)        (tensor* output, const tensor** inputs, const extent num_in, const void* params, const extent num_param);
typedef boolean constraint_t    (tensor* output, const tensor** inputs, const extent num_in, const void* params, const extent num_param);


tensor* tensor_alloc            (extent rank, const extent* shape, dtype_t dtype, mem_loc_t loc);    // error - error tensor                                      // error - error tensor
tensor* tensor_view_from        (tensor* t, const dptr* data, mem_loc_t loc);                        // error - error tensor
boolean tensor_free             (tensor* t);                                                         // error - 1
dptr*   tensor_view_to          (const tensor* t, mem_loc_t loc);                                    // error - null


/*
 * Primitive Kernel Set
**/
// 1. Memory
extern ker_t fill_const_ker;   extern constraint_t fill_const_constr;
extern ker_t copy_ker;         extern constraint_t copy_constr;
extern ker_t cast_ker;         extern constraint_t cast_constr;
extern ker_t reshape_ker;      extern constraint_t reshape_constr;
extern ker_t permute_ker;      extern constraint_t permute_constr;
extern ker_t slice_ker;        extern constraint_t slice_constr;
extern ker_t expand_ker;       extern constraint_t expand_constr;

// 2. Arithmetic (Elementwise)
// Unary
extern ker_t neg_ker;          extern constraint_t neg_constr;
extern ker_t abs_ker;          extern constraint_t abs_constr;
// Binary
extern ker_t add_ker;          extern constraint_t add_constr;
extern ker_t sub_ker;          extern constraint_t sub_constr;
extern ker_t mul_ker;          extern constraint_t mul_constr;
extern ker_t div_ker;          extern constraint_t div_constr;
// Ternary (only for real)
extern ker_t fma_ker;          extern constraint_t fma_constr;
extern ker_t fms_ker;          extern constraint_t fms_constr;
extern ker_t fnma_ker;         extern constraint_t fnma_constr;
extern ker_t fnms_ker;         extern constraint_t fnms_constr;

// 3. Comparisons (output is UINT8 mask)
extern ker_t eq_ker;           extern constraint_t eq_constr;
extern ker_t neq_ker;          extern constraint_t neq_constr;
extern ker_t gt_ker;           extern constraint_t gt_constr;
extern ker_t gte_ker;          extern constraint_t gte_constr;
extern ker_t lt_ker;           extern constraint_t lt_constr;
extern ker_t lte_ker;          extern constraint_t lte_constr;

// 4. Bitwise (Integer Only)
// Unary
extern ker_t bit_not_ker;      extern constraint_t bit_not_constr;
// Binary
extern ker_t bit_and_ker;      extern constraint_t bit_and_constr;
extern ker_t bit_or_ker;       extern constraint_t bit_or_constr;
extern ker_t bit_xor_ker;      extern constraint_t bit_xor_constr;
// Shifts
extern ker_t shl_ker;          extern constraint_t shl_constr;
extern ker_t shr_ker;          extern constraint_t shr_constr;
extern ker_t shr_u_ker;        extern constraint_t shr_u_constr;

// 5. Real-only
// Unary
extern ker_t sqrt_ker;         extern constraint_t sqrt_constr;
extern ker_t rsqrt_ker;        extern constraint_t rsqrt_constr;
extern ker_t recip_ker;        extern constraint_t recip_constr;
extern ker_t floor_ker;        extern constraint_t floor_constr;
extern ker_t ceil_ker;         extern constraint_t ceil_constr;
extern ker_t trunc_ker;        extern constraint_t trunc_constr;
extern ker_t round_ker;        extern constraint_t round_constr;
// Binary
extern ker_t min_ker;          extern constraint_t min_constr;
extern ker_t max_ker;          extern constraint_t max_constr;

// 6. Control / Selection
// select(mask, a, b) → out[i] = mask[i] ? a[i] : b[i]
extern ker_t sel_ker;          extern constraint_t sel_constr;

// 7. Reductions (RDC)
// Along a specified axis.
extern ker_t reduce_sum_ker;    extern constraint_t reduce_sum_constr;
extern ker_t reduce_prod_ker;   extern constraint_t reduce_prod_constr;
extern ker_t reduce_min_ker;    extern constraint_t reduce_min_constr;
extern ker_t reduce_max_ker;    extern constraint_t reduce_max_constr;
extern ker_t reduce_argmin_ker; extern constraint_t reduce_argmin_constr;
extern ker_t reduce_argmax_ker; extern constraint_t reduce_argmax_constr;
extern ker_t reduce_mean_ker;   extern constraint_t reduce_mean_constr;
// Logical reductions (UINT8 output)
extern ker_t reduce_all_ker;    extern constraint_t reduce_all_constr;
extern ker_t reduce_any_ker;    extern constraint_t reduce_any_constr;

#endif /* TENSOR_H */
