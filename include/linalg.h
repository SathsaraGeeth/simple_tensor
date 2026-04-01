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