#include "ker_util_sc_base.h"

/* ─── 2. Arithmetic ─── */
UNARY_OP(neg, any_type, DISPATCH_ANY_UNARY, -_a[_i])
// abs uses a type-aware ternary to avoid unsigned < 0 warning
UNARY_OP(abs, any_type, DISPATCH_ANY_UNARY, ((output->dtype >= UINT8) ? _a[_i] : (_a[_i] < 0 ? -_a[_i] : _a[_i])))

BINARY_OP(add, any_type, DISPATCH_ANY_BINARY, _a[_i] + _b[_i])
BINARY_OP(sub, any_type, DISPATCH_ANY_BINARY, _a[_i] - _b[_i])
BINARY_OP(mul, any_type, DISPATCH_ANY_BINARY, _a[_i] * _b[_i])
BINARY_OP(div, any_type, DISPATCH_ANY_BINARY, _a[_i] / _b[_i])

TERNARY_REAL_OP(fma,  (_a[_i] * _b[_i]) + _c[_i])
TERNARY_REAL_OP(fms,  (_a[_i] * _b[_i]) - _c[_i])
TERNARY_REAL_OP(fnma, -(_a[_i] * _b[_i]) + _c[_i])
TERNARY_REAL_OP(fnms, -(_a[_i] * _b[_i]) - _c[_i])

/* ─── 3. Comparisons ─── */
COMPARE_OP(eq,  _a[_i] == _b[_i])
COMPARE_OP(neq, _a[_i] != _b[_i])
COMPARE_OP(gt,  _a[_i] >  _b[_i])
COMPARE_OP(gte, _a[_i] >= _b[_i])
COMPARE_OP(lt,  _a[_i] <  _b[_i])
COMPARE_OP(lte, _a[_i] <= _b[_i])

/* ─── 4. Bitwise (Integer Only) ─── */
UNARY_OP(bit_not, int_only, DISPATCH_INT_ONLY_UNARY, ~_a[_i])
BINARY_OP(bit_and, int_only, DISPATCH_INT_ONLY_BINARY, _a[_i] & _b[_i])
BINARY_OP(bit_or,  int_only, DISPATCH_INT_ONLY_BINARY, _a[_i] | _b[_i])
BINARY_OP(bit_xor, int_only, DISPATCH_INT_ONLY_BINARY, _a[_i] ^ _b[_i])
BINARY_OP(shl,     int_only, DISPATCH_INT_ONLY_BINARY, _a[_i] << _b[_i])
BINARY_OP(shr,     int_only, DISPATCH_INT_ONLY_BINARY, _a[_i] >> _b[_i])

/* ─── 5. Real-only ─── */
UNARY_OP(sqrt,  real_only, DISPATCH_REAL_ONLY_UNARY, sqrt(_a[_i]))
UNARY_OP(rsqrt, real_only, DISPATCH_REAL_ONLY_UNARY, 1.0f / sqrt(_a[_i]))
UNARY_OP(recip, real_only, DISPATCH_REAL_ONLY_UNARY, 1.0f / _a[_i])
UNARY_OP(floor, real_only, DISPATCH_REAL_ONLY_UNARY, floor(_a[_i]))
UNARY_OP(ceil,  real_only, DISPATCH_REAL_ONLY_UNARY, ceil(_a[_i]))
UNARY_OP(trunc, real_only, DISPATCH_REAL_ONLY_UNARY, trunc(_a[_i]))
UNARY_OP(round, real_only, DISPATCH_REAL_ONLY_UNARY, round(_a[_i]))

BINARY_OP(min, real_only, DISPATCH_REAL_ONLY_BINARY, (_a[_i] < _b[_i]) ? _a[_i] : _b[_i])
BINARY_OP(max, real_only, DISPATCH_REAL_ONLY_BINARY, (_a[_i] > _b[_i]) ? _a[_i] : _b[_i])

/* 6. selcet */
SELECT_OP(sel)


/* --- 7. Reductions --- */
#define SUM(a,b) (a+b)
#define PROD(a,b) (a*b)
#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)
#define STEP(r,d) (r)
#define MEAN(r,d) (r/(float)d)

REDUCE_OP(reduce_sum,  0, SUM,  STEP)
REDUCE_OP(reduce_prod, 1, PROD, STEP)
REDUCE_OP(reduce_min,  INFINITY, MIN, STEP)
REDUCE_OP(reduce_max, -INFINITY, MAX, STEP)
REDUCE_OP(reduce_mean, 0, SUM,  MEAN)

static tensor* arg_impl(tensor* out, const tensor** ins, extent nin, const void* p, extent np, boolean is_max) {
    extent axis = ((extent*)p)[0]; const tensor* in = ins[0];
    extent dim = ((extent*)in->shape->ptr)[axis];
    tensor_prepare_output(out, in->size / dim);
    extent outer = 1; for(extent i=0; i<axis; i++) outer *= ((extent*)in->shape->ptr)[i];
    extent inner = 1; for(extent i=axis+1; i<in->rank; i++) inner *= ((extent*)in->shape->ptr)[i];
    int32* _o = (int32*)out->data->ptr; float* _i = (float*)in->data->ptr;
    for(extent o=0; o<outer; o++) { for(extent k=0; k<inner; k++) {
        float bv = is_max ? -INFINITY : INFINITY; int32 bi = 0;
        for(extent d=0; d<dim; d++) {
            float v = _i[(o*dim*inner)+(d*inner)+k];
            if(is_max ? (v>bv) : (v<bv)) { bv=v; bi=(int32)d; }
        }
        _o[o*inner+k] = bi;
    } } return out;
}
tensor* reduce_argmax_impl(tensor* o, const tensor** i, extent ni, const void* p, extent np) { return arg_impl(o,i,ni,p,np,true); }
tensor* reduce_argmin_impl(tensor* o, const tensor** i, extent ni, const void* p, extent np) { return arg_impl(o,i,ni,p,np,false); }
ker_t reduce_argmax_ker = reduce_argmax_impl;
ker_t reduce_argmin_ker = reduce_argmin_impl;

static tensor* logic_impl(tensor* out, const tensor** ins, extent nin, const void* p, extent np, boolean is_all) {
    extent axis = ((extent*)p)[0]; const tensor* in = ins[0];
    extent dim = ((extent*)in->shape->ptr)[axis];
    tensor_prepare_output(out, in->size / dim);
    extent outer = 1; for(extent i=0; i<axis; i++) outer *= ((extent*)in->shape->ptr)[i];
    extent inner = 1; for(extent i=axis+1; i<in->rank; i++) inner *= ((extent*)in->shape->ptr)[i];
    uint8* _o = (uint8*)out->data->ptr; float* _i = (float*)in->data->ptr;
    for(extent o=0; o<outer; o++) { for(extent k=0; k<inner; k++) {
        uint8 res = is_all ? 1 : 0;
        for(extent d=0; d<dim; d++) {
            boolean v = _i[(o*dim*inner)+(d*inner)+k] != 0;
            if(is_all) res &= v; else res |= v;
        }
        _o[o*inner+k] = res;
    } } return out;
}
tensor* reduce_all_impl(tensor* o, const tensor** i, extent ni, const void* p, extent np) { return logic_impl(o,i,ni,p,np,true); }
tensor* reduce_any_impl(tensor* o, const tensor** i, extent ni, const void* p, extent np) { return logic_impl(o,i,ni,p,np,false); }
ker_t reduce_all_ker = reduce_all_impl;
ker_t reduce_any_ker = reduce_any_impl;


/*
* fill_const
* Inputs: t0
* Output: t     (t0=t or t0!=t)
* Parmas: {const, dtype}
*
* Constraints:
* 1. Output must be owned.
* 2. The output->dtype must match the constant's dtype.
**/
static tensor* fill_const_impl(tensor* output, const tensor** inputs, const extent num_in, const void* params, const extent num_param) {
    (void)inputs; (void)num_in;
    if (!tensor_valid(output) || !output->data->is_owner || !params) return &ERROR_TENSOR;
    extent elem_size = dtype_size(output->dtype);
    if (num_param != elem_size) return &ERROR_TENSOR;
    uint8* dst = (uint8*)output->data->ptr;
    for (extent i = 0; i < output->size; i++) memcpy(dst + i * elem_size, params, elem_size);
    return output;
}
ker_t fill_const_ker = fill_const_impl;


/*
* copy
* Inputs: t0
* Output: t     (t0=t or t0!=t)
* Parmas: {const, dtype}
*
* Constraints:
* 1. Output must be owned.
* 2. The output->dtype must match the input->dtype.
* 3. output and input must have matching shapes.
* 4. input does not have to contiguous. (this method is used make a tensor non-contigous).
* 5. the location of input and output can be different. (this method is used to move tensor around devices).
**/



/*
* cast
* Input: N/A
* Output: t
* Params: {dtype}
*
* Constraints:
* 1. Output must be owned.
* 2. Output must be contigous.
**/

