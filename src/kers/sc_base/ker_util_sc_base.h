#ifndef KER_UTIL_SC_BASE_H
#define KER_UTIL_SC_BASE_H

#include "tensor.h"
#include <string.h>
#include <math.h>

/* ─── Verification ───────────────────────────────────────────────────────── */

static inline boolean tensor_valid(const tensor* t) {
    return t && !t->error && t->data && t->data->ptr;
}

static inline boolean tensors_shape_match(const tensor* a, const tensor* b) {
    if (a->rank != b->rank) return false;
    const extent* sa = (const extent*)a->shape->ptr;
    const extent* sb = (const extent*)b->shape->ptr;
    for (extent i = 0; i < a->rank; i++)
        if (sa[i] != sb[i]) return false;
    return true;
}

static inline boolean real_only(dtype_t type) { return (type == REAL32 || type == REAL64); }
static inline boolean int_only(dtype_t type)  { return (type >= INT8 && type <= UINT64); }
static inline boolean any_type(dtype_t type)  { (void)type; return true; }

/* ─── Specialized Dispatch Macros ────────────────────────────────────────── */

#define DISPATCH_INT_ONLY_UNARY(output, t0, EXPR) \
    switch ((output)->dtype) { \
        case INT8:  { int8* _o=(int8*)(output)->data->ptr; const int8* _a=(int8*)(t0)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case INT16: { int16* _o=(int16*)(output)->data->ptr; const int16* _a=(int16*)(t0)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case INT32: { int32* _o=(int32*)(output)->data->ptr; const int32* _a=(int32*)(t0)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case INT64: { int64* _o=(int64*)(output)->data->ptr; const int64* _a=(int64*)(t0)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case UINT8: { uint8* _o=(uint8*)(output)->data->ptr; const uint8* _a=(uint8*)(t0)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case UINT16:{ uint16* _o=(uint16*)(output)->data->ptr; const uint16* _a=(uint16*)(t0)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case UINT32:{ uint32* _o=(uint32*)(output)->data->ptr; const uint32* _a=(uint32*)(t0)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case UINT64:{ uint64* _o=(uint64*)(output)->data->ptr; const uint64* _a=(uint64*)(t0)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        default: return &ERROR_TENSOR; \
    }

#define DISPATCH_REAL_ONLY_UNARY(output, t0, EXPR) \
    switch ((output)->dtype) { \
        case REAL32:{ real32* _o=(real32*)(output)->data->ptr; const real32* _a=(real32*)(t0)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case REAL64:{ real64* _o=(real64*)(output)->data->ptr; const real64* _a=(real64*)(t0)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        default: return &ERROR_TENSOR; \
    }

#define DISPATCH_ANY_UNARY(output, t0, EXPR) \
    switch ((output)->dtype) { \
        case REAL32: case REAL64: DISPATCH_REAL_ONLY_UNARY(output, t0, EXPR) break; \
        default: DISPATCH_INT_ONLY_UNARY(output, t0, EXPR) break; \
    }

#define DISPATCH_INT_ONLY_BINARY(output, t0, t1, EXPR) \
    switch ((output)->dtype) { \
        case INT8:  { int8* _o=(int8*)(output)->data->ptr; const int8* _a=(int8*)(t0)->data->ptr; const int8* _b=(int8*)(t1)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case INT16: { int16* _o=(int16*)(output)->data->ptr; const int16* _a=(int16*)(t0)->data->ptr; const int16* _b=(int16*)(t1)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case INT32: { int32* _o=(int32*)(output)->data->ptr; const int32* _a=(int32*)(t0)->data->ptr; const int32* _b=(int32*)(t1)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case INT64: { int64* _o=(int64*)(output)->data->ptr; const int64* _a=(int64*)(t0)->data->ptr; const int64* _b=(int64*)(t1)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case UINT8: { uint8* _o=(uint8*)(output)->data->ptr; const uint8* _a=(uint8*)(t0)->data->ptr; const uint8* _b=(uint8*)(t1)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case UINT16:{ uint16* _o=(uint16*)(output)->data->ptr; const uint16* _a=(uint16*)(t0)->data->ptr; const uint16* _b=(uint16*)(t1)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case UINT32:{ uint32* _o=(uint32*)(output)->data->ptr; const uint32* _a=(uint32*)(t0)->data->ptr; const uint32* _b=(uint32*)(t1)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case UINT64:{ uint64* _o=(uint64*)(output)->data->ptr; const uint64* _a=(uint64*)(t0)->data->ptr; const uint64* _b=(uint64*)(t1)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        default: return &ERROR_TENSOR; \
    }

#define DISPATCH_REAL_ONLY_BINARY(output, t0, t1, EXPR) \
    switch ((output)->dtype) { \
        case REAL32:{ real32* _o=(real32*)(output)->data->ptr; const real32* _a=(real32*)(t0)->data->ptr; const real32* _b=(real32*)(t1)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        case REAL64:{ real64* _o=(real64*)(output)->data->ptr; const real64* _a=(real64*)(t0)->data->ptr; const real64* _b=(real64*)(t1)->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(EXPR); break; } \
        default: return &ERROR_TENSOR; \
    }

#define DISPATCH_ANY_BINARY(output, t0, t1, EXPR) \
    switch ((output)->dtype) { \
        case REAL32: case REAL64: DISPATCH_REAL_ONLY_BINARY(output, t0, t1, EXPR) break; \
        default: DISPATCH_INT_ONLY_BINARY(output, t0, t1, EXPR) break; \
    }

/* ─── Kernel Templates ───────────────────────────────────────────────────── */

#define UNARY_OP(name, type_check, dispatch_macro, expr) \
static tensor* name##_impl(tensor* output, const tensor** inputs, const extent num_in, const void* params, const extent num_param) { \
    (void)params; (void)num_param; \
    if (num_in != 1 || !tensor_valid(inputs[0]) || !inputs[0]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(output) || !output->is_owned || !output->is_contiguous) return &ERROR_TENSOR; \
    if (inputs[0]->dtype != output->dtype || !type_check(output->dtype)) return &ERROR_TENSOR; \
    if (!tensors_shape_match(inputs[0], output)) return &ERROR_TENSOR; \
    extent _n = output->size; \
    dispatch_macro(output, inputs[0], expr); \
    return output; \
} \
ker_t name##_ker = name##_impl;

#define BINARY_OP(name, type_check, dispatch_macro, expr) \
static tensor* name##_impl(tensor* output, const tensor** inputs, const extent num_in, const void* params, const extent num_param) { \
    (void)params; (void)num_param; \
    if (num_in != 2 || !tensor_valid(inputs[0]) || !inputs[0]->is_contiguous || !tensor_valid(inputs[1]) || !inputs[1]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(output) || !output->is_owned || !output->is_contiguous) return &ERROR_TENSOR; \
    if (inputs[0]->dtype != inputs[1]->dtype || inputs[0]->dtype != output->dtype || !type_check(output->dtype)) return &ERROR_TENSOR; \
    if (!tensors_shape_match(inputs[0], inputs[1]) || !tensors_shape_match(inputs[0], output)) return &ERROR_TENSOR; \
    extent _n = output->size; \
    dispatch_macro(output, inputs[0], inputs[1], expr); \
    return output; \
} \
ker_t name##_ker = name##_impl;

/* Compare and Ternary handled similarly but specialized... */
#define TERNARY_REAL_OP(name, expr) \
static tensor* name##_impl(tensor* output, const tensor** inputs, const extent num_in, const void* params, const extent num_param) { \
    (void)params; (void)num_param; \
    if (num_in != 3 || !real_only(output->dtype)) return &ERROR_TENSOR; \
    if (!tensor_valid(inputs[0]) || !inputs[0]->is_contiguous || !tensor_valid(inputs[1]) || !inputs[1]->is_contiguous || !tensor_valid(inputs[2]) || !inputs[2]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(output) || !output->is_owned || !output->is_contiguous) return &ERROR_TENSOR; \
    if (inputs[0]->dtype != output->dtype || inputs[1]->dtype != output->dtype || inputs[2]->dtype != output->dtype) return &ERROR_TENSOR; \
    if (!tensors_shape_match(inputs[0], output) || !tensors_shape_match(inputs[1], output) || !tensors_shape_match(inputs[2], output)) return &ERROR_TENSOR; \
    extent _n = output->size; \
    switch (output->dtype) { \
        case REAL32:{ real32* _o=(real32*)(output)->data->ptr; const real32* _a=(real32*)(inputs[0])->data->ptr; const real32* _b=(real32*)(inputs[1])->data->ptr; const real32* _c=(real32*)(inputs[2])->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(expr); break; } \
        case REAL64:{ real64* _o=(real64*)(output)->data->ptr; const real64* _a=(real64*)(inputs[0])->data->ptr; const real64* _b=(real64*)(inputs[1])->data->ptr; const real64* _c=(real64*)(inputs[2])->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(expr); break; } \
        default: return &ERROR_TENSOR; \
    } \
    return output; \
} \
ker_t name##_ker = name##_impl;

#define COMPARE_OP(name, expr) \
static tensor* name##_impl(tensor* output, const tensor** inputs, const extent num_in, const void* params, const extent num_param) { \
    (void)params; (void)num_param; \
    if (num_in != 2 || !tensor_valid(inputs[0]) || !inputs[0]->is_contiguous || !tensor_valid(inputs[1]) || !inputs[1]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(output) || !output->is_owned || !output->is_contiguous || output->dtype != UINT8) return &ERROR_TENSOR; \
    if (inputs[0]->dtype != inputs[1]->dtype) return &ERROR_TENSOR; \
    if (!tensors_shape_match(inputs[0], inputs[1]) || !tensors_shape_match(inputs[0], output)) return &ERROR_TENSOR; \
    extent _n = output->size; uint8* _o = (uint8*)output->data->ptr; \
    switch(inputs[0]->dtype) { \
        case INT8:   { const int8* _a=(int8*)inputs[0]->data->ptr; const int8* _b=(int8*)inputs[1]->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(expr); break; } \
        case INT32:  { const int32* _a=(int32*)inputs[0]->data->ptr; const int32* _b=(int32*)inputs[1]->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(expr); break; } \
        case REAL32: { const real32* _a=(real32*)inputs[0]->data->ptr; const real32* _b=(real32*)inputs[1]->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(expr); break; } \
        case REAL64: { const real64* _a=(real64*)inputs[0]->data->ptr; const real64* _b=(real64*)inputs[1]->data->ptr; for(extent _i=0;_i<_n;_i++) _o[_i]=(expr); break; } \
        default: return &ERROR_TENSOR; \
    } \
    return output; \
} \
ker_t name##_ker = name##_impl;

/* Add this to your Kernel Templates section in ker_util_sc_base.h */

#define SELECT_OP(name) \
static tensor* name##_impl(tensor* output, const tensor** inputs, const extent num_in, const void* params, const extent num_param) { \
    (void)params; (void)num_param; \
    if (num_in != 3) return &ERROR_TENSOR; \
    /* inputs[0] is mask, inputs[1] is 'then', inputs[2] is 'else' */ \
    if (!tensor_valid(inputs[0]) || inputs[0]->dtype != UINT8 || !inputs[0]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(inputs[1]) || !inputs[1]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(inputs[2]) || !inputs[2]->is_contiguous) return &ERROR_TENSOR; \
    if (!tensor_valid(output)    || !output->is_owned      || !output->is_contiguous) return &ERROR_TENSOR; \
    if (inputs[1]->dtype != output->dtype || inputs[2]->dtype != output->dtype) return &ERROR_TENSOR; \
    if (!tensors_shape_match(inputs[0], output) || !tensors_shape_match(inputs[1], output) || !tensors_shape_match(inputs[2], output)) return &ERROR_TENSOR; \
    extent _n = output->size; \
    const uint8* _m = (uint8*)inputs[0]->data->ptr; \
    switch (output->dtype) { \
        case INT8:   { int8* _o=(int8*)output->data->ptr;   const int8* _a=(int8*)inputs[1]->data->ptr;   const int8* _b=(int8*)inputs[2]->data->ptr;   for(extent i=0;i<_n;i++) _o[i]=_m[i]?_a[i]:_b[i]; break; } \
        case INT32:  { int32* _o=(int32*)output->data->ptr;  const int32* _a=(int32*)inputs[1]->data->ptr;  const int32* _b=(int32*)inputs[2]->data->ptr;  for(extent i=0;i<_n;i++) _o[i]=_m[i]?_a[i]:_b[i]; break; } \
        case REAL32: { real32* _o=(real32*)output->data->ptr; const real32* _a=(real32*)inputs[1]->data->ptr; const real32* _b=(real32*)inputs[2]->data->ptr; for(extent i=0;i<_n;i++) _o[i]=_m[i]?_a[i]:_b[i]; break; } \
        case REAL64: { real64* _o=(real64*)output->data->ptr; const real64* _a=(real64*)inputs[1]->data->ptr; const real64* _b=(real64*)inputs[2]->data->ptr; for(extent i=0;i<_n;i++) _o[i]=_m[i]?_a[i]:_b[i]; break; } \
        default: return &ERROR_TENSOR; \
    } \
    return output; \
} \
ker_t name##_ker = name##_impl;


static inline boolean tensor_prepare_output(tensor* t, extent req_size) {
    if (!t || !t->is_owned || !t->data) return false; 
    extent bytes = req_size * dtype_size(t->dtype);
    if (t->data->size != bytes) {
        mem_loc_t loc = t->data->loc;
        mem_algn_t alg = t->data->alignment;
        mem_free(t->data);
        t->data = mem_alloc(bytes, loc);
        if (!t->data) { t->error = true; return false; }
        t->data->alignment = alg;
        t->size = req_size;
        t->is_contiguous = true;
    }
    return true;
}

#define REDUCE_OP(name, INIT, OP, FINAL) \
static tensor* name##_impl(tensor* out, const tensor** ins, extent nin, const void* p, extent np) { \
    if (nin != 1 || !p || np < sizeof(extent)) return &ERROR_TENSOR; \
    extent axis = ((extent*)p)[0]; const tensor* in = ins[0]; \
    extent dim = ((extent*)in->shape->ptr)[axis]; \
    if (!tensor_prepare_output(out, in->size / dim)) return &ERROR_TENSOR; \
    extent outer = 1; for(extent i=0; i<axis; i++) outer *= ((extent*)in->shape->ptr)[i]; \
    extent inner = 1; for(extent i=axis+1; i<in->rank; i++) inner *= ((extent*)in->shape->ptr)[i]; \
    float* _o = (float*)out->data->ptr; const float* _i = (float*)in->data->ptr; \
    for(extent o=0; o<outer; o++) { for(extent k=0; k<inner; k++) { \
        float res = (float)(INIT); \
        for(extent d=0; d<dim; d++) res = OP(res, _i[(o*dim*inner)+(d*inner)+k]); \
        _o[o*inner+k] = FINAL(res, dim); \
    } } return out; \
} \
ker_t name##_ker = name##_impl;

#endif