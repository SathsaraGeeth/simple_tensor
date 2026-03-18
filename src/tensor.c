// #include <string.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <stdbool.h>
// #include <stdio.h>
// #include <sys/types.h>
// #include "tensor.h"
// #include <sys/types.h>

// // 0. Internal Helpers
// static size_t compute_numel(size_t rank, const size_t* shape) {
//     size_t n = 1;
//     for (size_t i = 0; i < rank; ++i) n *= shape[i];
//     return n;
// }
// static size_t broadcast_prepare(
//     const Tensor* a, const Tensor* b,
//     size_t** out_shape,
//     size_t** out_stride_a,
//     size_t** out_stride_b
// ) {
//     size_t rank = (a->rank > b->rank) ? a->rank : b->rank;
//     *out_shape    = malloc(rank * sizeof(size_t));
//     *out_stride_a = malloc(rank * sizeof(size_t));
//     *out_stride_b = malloc(rank * sizeof(size_t));

//     for (ssize_t i = rank - 1, ia = a->rank - 1, ib = b->rank - 1;
//          i >= 0; --i, --ia, --ib) {
//         size_t dim_a = (ia >= 0) ? a->shape[ia] : 1;
//         size_t dim_b = (ib >= 0) ? b->shape[ib] : 1;
//         if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
//             fprintf(stderr, "Broadcast shape mismatch\n");
//             exit(EXIT_FAILURE);
//         }
//         size_t dim = (dim_a > dim_b) ? dim_a : dim_b;
//         (*out_shape)[i]    = dim;
//         (*out_stride_a)[i] = (ia >= 0 && dim_a != 1) ? a->strides[ia] : 0;
//         (*out_stride_b)[i] = (ib >= 0 && dim_b != 1) ? b->strides[ib] : 0;
//     }
//     return rank;
// }




// // 4. Utility Functions: _util
// Tensor* tensor_util_copy(const Tensor* t) {
//     tensor_util_assert_valid(t);
//     Tensor* out = tensor_util_like(t);
//     size_t bytes = t->size * tensor_util_dtype_size(t->dtype);
//     memcpy(out->data, t->data, bytes);
//     return out;
// }
// void tensor_util_print(const Tensor* t) {
//     tensor_util_assert_valid(t);
//     printf("Tensor(shape=[");
//     for (size_t i = 0; i < t->rank; ++i) {
//         printf("%zu", t->shape[i]);
//         if (i < t->rank - 1) printf(", ");
//     }
//     printf("], dtype=%d)\n", t->dtype);
// }
// size_t tensor_util_index(const Tensor* t, const size_t* indices) {
//     tensor_util_assert_valid(t);
//     size_t offset = 0;
//     for (size_t i = 0; i < t->rank; ++i) offset += indices[i] * t->strides[i];
//     return offset;
// }
// size_t tensor_util_dtype_size(DTypes dtype) {
//     switch (dtype) {
//         case FP64:  return 8;
//         case FP32:  return 4;
//         case INT64: return 8;
//         case INT32: return 4;
//         case INT16: return 2;
//         case INT8:  return 1;
//         case UINT64:return 8;
//         case UINT32:return 4;
//         case UINT16:return 2;
//         case UINT8: return 1;
//         default:
//             fprintf(stderr, "Unknown dtype\n");
//             exit(EXIT_FAILURE);
//     }
// }
// size_t tensor_util_numel(const Tensor* t) { return t->size; }
// void tensor_util_compute_strides(Tensor* t) {
//     t->strides[t->rank - 1] = 1;
//     for (ssize_t i = (ssize_t)t->rank - 2; i >= 0; --i)
//         t->strides[i] = t->strides[i+1] * t->shape[i+1];
// }
// bool tensor_util_is_contiguous(const Tensor* t) {
//     size_t e = 1;
//     for (ssize_t i = (ssize_t)t->rank - 1; i >= 0; --i) {
//         if (t->strides[i] != e) return false;
//         e *= t->shape[i];
//     }
//     return true;
// }
// size_t tensor_util_offset(const Tensor* t, const size_t* indices) {
//     size_t offset = 0;
//     for (size_t i = 0; i < t->rank; ++i) offset += indices[i] * t->strides[i];
//     return offset;
// }
// void* tensor_util_data_ptr(Tensor* t, const size_t* indices) {
//     return (uint8_t*)t->data + tensor_util_offset(t, indices) * tensor_util_dtype_size(t->dtype);
// }
// const void* tensor_util_const_data_ptr(const Tensor* t, const size_t* indices) {
//     return (const uint8_t*)t->data + tensor_util_offset(t, indices) * tensor_util_dtype_size(t->dtype);
// }
// void tensor_util_assert_valid(const Tensor* t) {
//     if (!t || !t->shape || !t->strides) {
//         fprintf(stderr, "Invalid Tensor\n");
//         exit(EXIT_FAILURE);
//     }
// }
// Tensor* tensor_util_like(const Tensor* t) {
//     return tensor_mem_alloc(t->rank, t->shape, t->dtype);
// }








// // 1. Memory Management: _mem
// Tensor* tensor_mem_alloc(size_t rank, const size_t* shape, DTypes dtype) {
//     Tensor* t = malloc(sizeof(Tensor));
//     if (!t) exit(EXIT_FAILURE);
//     t->rank = rank;
//     t->dtype = dtype;
//     t->shape = malloc(rank * sizeof(size_t));
//     t->strides = malloc(rank * sizeof(size_t));
//     for (size_t i = 0; i < rank; ++i) t->shape[i] = shape[i];
//     t->size = compute_numel(rank, shape);
//     tensor_util_compute_strides(t);
//     t->data = malloc(t->size * tensor_util_dtype_size(dtype));
//     return t;
// }
// void tensor_mem_free(Tensor* t) {
//     if (!t) return;
//     free(t->data);
//     free(t->shape);
//     free(t->strides);
//     free(t);
// }





// // 2. Metadata Access: _meta
// size_t tensor_meta_rank(const Tensor* t) { return t->rank; }
// size_t tensor_meta_size(const Tensor* t) { return t->size; }
// DTypes tensor_meta_dtype(const Tensor* t) { return t->dtype; }
// const size_t* tensor_meta_shape(const Tensor* t) { return t->shape; }
// const size_t* tensor_meta_strides(const Tensor* t) { return t->strides; }




// // 3.1 View Operations: _view
// Tensor* tensor_op_view_reshape(Tensor* t, size_t new_rank, const size_t* new_shape) {
//     tensor_util_assert_valid(t);
//     if (compute_numel(new_rank, new_shape) != t->size) exit(EXIT_FAILURE);
//     Tensor* out = malloc(sizeof(Tensor));
//     *out = *t;
//     out->rank = new_rank;
//     out->shape = malloc(new_rank * sizeof(size_t));
//     out->strides = malloc(new_rank * sizeof(size_t));
//     for (size_t i = 0; i < new_rank; ++i) out->shape[i] = new_shape[i];
//     tensor_util_compute_strides(out);
//     return out;
// }
// Tensor* tensor_op_view_flatten(Tensor* t) {
//     size_t s[1] = {t->size};
//     return tensor_op_view_reshape(t, 1, s);
// }





// // 3.2 Elementwise Operations: _ew
// Tensor* tensor_op_ew_add(const Tensor* a, const Tensor* b) {
//     tensor_util_assert_valid(a);
//     tensor_util_assert_valid(b);
//     if (a->dtype != b->dtype || a->dtype != FP32) exit(EXIT_FAILURE);
//     size_t* shape; size_t* sa; size_t* sb;
//     size_t rank = broadcast_prepare(a, b, &shape, &sa, &sb);
//     Tensor* out = tensor_mem_alloc(rank, shape, a->dtype);
//     const float* da = a->data;
//     const float* db = b->data;
//     float* o = out->data;
//     size_t* idx = calloc(rank, sizeof(size_t));
//     for (size_t i = 0; i < out->size; ++i) {
//         size_t off_a = 0;
//         size_t off_b = 0;
//         for (size_t d = 0; d < rank; ++d) {
//             off_a += idx[d] * sa[d];
//             off_b += idx[d] * sb[d];
//         }
//         o[i] = da[off_a] + db[off_b];
//         for (ssize_t d = rank - 1; d >= 0; --d) {
//             idx[d]++;
//             if (idx[d] < shape[d]) break;
//             idx[d] = 0;
//         }
//     }
//     free(shape); free(sa); free(sb); free(idx);
//     return out;
// }






// // 3.4 Scalar Operations: _sc
// Tensor* tensor_op_sc_add(const Tensor* t, double val) {
//     tensor_util_assert_valid(t);
//     if (t->dtype != FP32) exit(EXIT_FAILURE);
//     Tensor* out = tensor_util_like(t);
//     float* restrict o = out->data;
//     const float* restrict d = t->data;
//     float v = (float)val;
//     for (size_t i = 0; i < t->size; ++i) o[i] = d[i] + v;
//     return out;
// }
// Tensor* tensor_op_sc_sub(const Tensor* t, double val) {
//     return tensor_op_sc_add(t, -val);
// }
// Tensor* tensor_op_sc_mul(const Tensor* t, double val) {
//     tensor_util_assert_valid(t);
//     if (t->dtype != FP32) exit(EXIT_FAILURE);
//     Tensor* out = tensor_util_like(t);
//     float* restrict o = out->data;
//     const float* restrict d = t->data;
//     float v = (float)val;
//     for (size_t i = 0; i < t->size; ++i) o[i] = d[i] * v;
//     return out;
// }
// Tensor* tensor_op_sc_div(const Tensor* t, double val) {
//     if (val == 0.0) exit(EXIT_FAILURE);
//     return tensor_op_sc_mul(t, 1.0 / val);
// }
// Tensor* tensor_op_sc_neg(const Tensor* t) {
//     return tensor_op_sc_mul(t, -1.0);
// }
// Tensor* tensor_op_sc_abs(const Tensor* t) {
//     tensor_util_assert_valid(t);
//     if (t->dtype != FP32) exit(EXIT_FAILURE);
//     Tensor* out = tensor_util_like(t);
//     float* restrict o = out->data;
//     const float* restrict d = t->data;
//     for (size_t i = 0; i < t->size; ++i) o[i] = (d[i] < 0) ? -d[i] : d[i];
//     return out;
// }











// // ===== NOT IMPLEMENTED STUBS =====
// #define NOT_IMPL(name) do { fprintf(stderr, name " not implemented\n"); exit(EXIT_FAILURE); } while(0)

// // View
// Tensor* tensor_op_view_broadcast(Tensor* t, size_t r, const size_t* s) { NOT_IMPL("view_broadcast"); }
// Tensor* tensor_op_view_squeeze(Tensor* t) { NOT_IMPL("view_squeeze"); }
// Tensor* tensor_op_view_unsqueeze(Tensor* t, size_t a) { NOT_IMPL("view_unsqueeze"); }
// Tensor* tensor_op_view_permute(Tensor* t, const size_t* a) { NOT_IMPL("view_permute"); }
// Tensor* tensor_op_view_concat(const Tensor** t, size_t n, size_t a) { NOT_IMPL("view_concat"); }

// // Elementwise
// Tensor* tensor_op_ew_sub(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_sub"); }
// Tensor* tensor_op_ew_mul(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_mul"); }
// Tensor* tensor_op_ew_div(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_div"); }
// Tensor* tensor_op_ew_pow(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_pow"); }
// Tensor* tensor_op_ew_min(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_min"); }
// Tensor* tensor_op_ew_max(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_max"); }
// Tensor* tensor_op_ew_eq(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_eq"); }
// Tensor* tensor_op_ew_neq(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_neq"); }
// Tensor* tensor_op_ew_gt(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_gt"); }
// Tensor* tensor_op_ew_gte(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_gte"); }
// Tensor* tensor_op_ew_lt(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_lt"); }
// Tensor* tensor_op_ew_lte(const Tensor* a,const Tensor* b){ NOT_IMPL("ew_lte"); }

// // Reduction
// Tensor* tensor_op_rd_sum(const Tensor* t,size_t a){ NOT_IMPL("rd_sum"); }
// Tensor* tensor_op_rd_mean(const Tensor* t,size_t a){ NOT_IMPL("rd_mean"); }
// Tensor* tensor_op_rd_max(const Tensor* t,size_t a){ NOT_IMPL("rd_max"); }
// Tensor* tensor_op_rd_min(const Tensor* t,size_t a){ NOT_IMPL("rd_min"); }
// Tensor* tensor_op_rd_argmax(const Tensor* t,size_t a){ NOT_IMPL("rd_argmax"); }
// Tensor* tensor_op_rd_argmin(const Tensor* t,size_t a){ NOT_IMPL("rd_argmin"); }

// // Linear Algebra
// Tensor* tensor_op_la_dot(const Tensor* a,const Tensor* b){ NOT_IMPL("la_dot"); }
// Tensor* tensor_op_la_matmul(const Tensor* a,const Tensor* b){ NOT_IMPL("la_matmul"); }
// Tensor* tensor_op_la_transpose(const Tensor* t,const size_t* a){ NOT_IMPL("la_transpose"); }
