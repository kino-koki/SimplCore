// Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co,.Ltd / 六安市青奈科技有限责任公司
// SimpCore C++ Extension Module
// High-performance tensor operations implementation

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#define SIMPCORE_VERSION "1.0.0"

static PyMethodDef SimpCoreMethods[] = {
    {"matmul", (PyCFunction)matmul_impl, METH_VARARGS, "Matrix multiplication"},
    {"softmax", (PyCFunction)softmax_impl, METH_VARARGS, "Softmax operation"},
    {"layer_norm", (PyCFunction)layer_norm_impl, METH_VARARGS, "Layer normalization"},
    {"conv2d", (PyCFunction)conv2d_impl, METH_VARARGS, "2D convolution"},
    {"relu", (PyCFunction)relu_impl, METH_VARARGS, "ReLU activation"},
    {"gelu", (PyCFunction)gelu_impl, METH_VARARGS, "GELU activation"},
    {"sigmoid", (PyCFunction)sigmoid_impl, METH_VARARGS, "Sigmoid activation"},
    {"tanh", (PyCFunction)tanh_impl, METH_VARARGS, "Tanh activation"},
    {"addmm", (PyCFunction)addmm_impl, METH_VARARGS, "Matrix multiplication with bias"},
    {"batch_norm", (PyCFunction)batch_norm_impl, METH_VARARGS, "Batch normalization"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef csimpcore_module = {
    PyModuleDef_HEAD_INIT,
    "csimpcore",
    "SimpCore C++ Extension Module - High-performance tensor operations",
    -1,
    SimpCoreMethods
};

PyMODINIT_FUNC PyInit_csimpcore(void) {
    import_array();
    return PyModule_Create(&csimpcore_module);
}

inline float relu_f(float x) { return x > 0 ? x : 0; }
inline float gelu_f(float x) { return x * 0.5f * (1.0f + tanh(x * 0.79788456f * (1.0f + 0.044715f * x * x))); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float tanh_f(float x) { return tanh(x); }

static PyObject* matmul_impl(PyObject* self, PyObject* args) {
    PyArrayObject *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a, &PyArray_Type, &b)) {
        return NULL;
    }

    if (PyArray_TYPE(a) != NPY_FLOAT32 || PyArray_TYPE(b) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "matmul requires float32 arrays");
        return NULL;
    }

    int ndim_a = PyArray_NDIM(a);
    int ndim_b = PyArray_NDIM(b);
    
    if (ndim_a != 2 || ndim_b != 2) {
        PyErr_SetString(PyExc_ValueError, "matmul requires 2D arrays");
        return NULL;
    }

    npy_intp *dims_a = PyArray_DIMS(a);
    npy_intp *dims_b = PyArray_DIMS(b);
    
    int m = (int)dims_a[0];
    int k = (int)dims_a[1];
    int n = (int)dims_b[1];

    if (dims_a[1] != dims_b[0]) {
        PyErr_SetString(PyExc_ValueError, "Matrix dimensions do not match");
        return NULL;
    }

    float *data_a = (float*)PyArray_DATA(a);
    float *data_b = (float*)PyArray_DATA(b);

    npy_intp dims_c[2] = {m, n};
    PyArrayObject *c = (PyArrayObject*)PyArray_SimpleNew(2, dims_c, NPY_FLOAT32);
    float *data_c = (float*)PyArray_DATA(c);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += data_a[i * k + p] * data_b[p * n + j];
            }
            data_c[i * n + j] = sum;
        }
    }

    return (PyObject*)c;
}

static PyObject* softmax_impl(PyObject* self, PyObject* args) {
    PyArrayObject *x;
    int dim = -1;
    if (!PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &x, &dim)) {
        return NULL;
    }

    if (PyArray_TYPE(x) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "softmax requires float32 arrays");
        return NULL;
    }

    int ndim = PyArray_NDIM(x);
    npy_intp *dims = PyArray_DIMS(x);
    
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        PyErr_SetString(PyExc_ValueError, "Invalid dimension");
        return NULL;
    }

    float *data = (float*)PyArray_DATA(x);
    npy_intp size = PyArray_SIZE(x);
    int dim_size = (int)dims[dim];
    
    npy_intp outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= dims[i];
    npy_intp inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) inner_size *= dims[i];

    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(ndim, dims, NPY_FLOAT32);
    float *result_data = (float*)PyArray_DATA(result);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int outer = 0; outer < outer_size; outer++) {
        for (int inner = 0; inner < inner_size; inner++) {
            int offset = outer * dim_size * inner_size + inner;
            float max_val = -INFINITY;
            for (int i = 0; i < dim_size; i++) {
                max_val = std::max(max_val, data[offset + i * inner_size]);
            }
            
            float sum = 0.0f;
            for (int i = 0; i < dim_size; i++) {
                float exp_val = exp(data[offset + i * inner_size] - max_val);
                sum += exp_val;
                result_data[offset + i * inner_size] = exp_val;
            }
            
            for (int i = 0; i < dim_size; i++) {
                result_data[offset + i * inner_size] /= sum;
            }
        }
    }

    return (PyObject*)result;
}

static PyObject* layer_norm_impl(PyObject* self, PyObject* args) {
    PyArrayObject *x, *weight, *bias;
    float eps = 1e-5f;
    
    if (!PyArg_ParseTuple(args, "O!O!O!|f", &PyArray_Type, &x, &PyArray_Type, &weight, &PyArray_Type, &bias, &eps)) {
        return NULL;
    }

    if (PyArray_TYPE(x) != NPY_FLOAT32 || PyArray_TYPE(weight) != NPY_FLOAT32 || PyArray_TYPE(bias) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "layer_norm requires float32 arrays");
        return NULL;
    }

    int ndim = PyArray_NDIM(x);
    npy_intp *dims = PyArray_DIMS(x);
    
    int last_dim = (int)dims[ndim - 1];
    npy_intp batch_size = PyArray_SIZE(x) / last_dim;

    float *data_x = (float*)PyArray_DATA(x);
    float *data_w = (float*)PyArray_DATA(weight);
    float *data_b = (float*)PyArray_DATA(bias);

    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(ndim, dims, NPY_FLOAT32);
    float *result_data = (float*)PyArray_DATA(result);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < batch_size; i++) {
        int offset = i * last_dim;
        float sum = 0.0f;
        for (int j = 0; j < last_dim; j++) {
            sum += data_x[offset + j];
        }
        float mean = sum / last_dim;
        
        float var_sum = 0.0f;
        for (int j = 0; j < last_dim; j++) {
            float diff = data_x[offset + j] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / last_dim;
        float std = sqrt(var + eps);
        
        for (int j = 0; j < last_dim; j++) {
            result_data[offset + j] = (data_x[offset + j] - mean) / std * data_w[j] + data_b[j];
        }
    }

    return (PyObject*)result;
}

static PyObject* conv2d_impl(PyObject* self, PyObject* args) {
    PyArrayObject *x, *weight, *bias;
    int stride_h = 1, stride_w = 1, padding_h = 0, padding_w = 0;
    
    if (!PyArg_ParseTuple(args, "O!O!O!|iiii", &PyArray_Type, &x, &PyArray_Type, &weight, &PyArray_Type, &bias, 
                          &stride_h, &stride_w, &padding_h, &padding_w)) {
        return NULL;
    }

    if (PyArray_TYPE(x) != NPY_FLOAT32 || PyArray_TYPE(weight) != NPY_FLOAT32 || PyArray_TYPE(bias) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "conv2d requires float32 arrays");
        return NULL;
    }

    npy_intp *dims_x = PyArray_DIMS(x);
    npy_intp *dims_w = PyArray_DIMS(weight);
    
    int batch = (int)dims_x[0];
    int in_channels = (int)dims_x[1];
    int in_h = (int)dims_x[2];
    int in_w = (int)dims_x[3];
    
    int out_channels = (int)dims_w[0];
    int kernel_h = (int)dims_w[2];
    int kernel_w = (int)dims_w[3];

    int out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;

    float *data_x = (float*)PyArray_DATA(x);
    float *data_w = (float*)PyArray_DATA(weight);
    float *data_b = (float*)PyArray_DATA(bias);

    npy_intp dims_out[4] = {batch, out_channels, out_h, out_w};
    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(4, dims_out, NPY_FLOAT32);
    float *result_data = (float*)PyArray_DATA(result);

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = data_b[oc];
                    int ih_start = oh * stride_h - padding_h;
                    int iw_start = ow * stride_w - padding_w;
                    
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int ih = ih_start + kh;
                            int iw = iw_start + kw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                for (int ic = 0; ic < in_channels; ic++) {
                                    float val_x = data_x[b * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw];
                                    float val_w = data_w[oc * in_channels * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw];
                                    sum += val_x * val_w;
                                }
                            }
                        }
                    }
                    result_data[b * out_channels * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
    }

    return (PyObject*)result;
}

static PyObject* relu_impl(PyObject* self, PyObject* args) {
    PyArrayObject *x;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x)) {
        return NULL;
    }

    if (PyArray_TYPE(x) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "relu requires float32 arrays");
        return NULL;
    }

    npy_intp size = PyArray_SIZE(x);
    float *data = (float*)PyArray_DATA(x);

    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_FLOAT32);
    float *result_data = (float*)PyArray_DATA(result);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (npy_intp i = 0; i < size; i++) {
        result_data[i] = relu_f(data[i]);
    }

    return (PyObject*)result;
}

static PyObject* gelu_impl(PyObject* self, PyObject* args) {
    PyArrayObject *x;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x)) {
        return NULL;
    }

    if (PyArray_TYPE(x) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "gelu requires float32 arrays");
        return NULL;
    }

    npy_intp size = PyArray_SIZE(x);
    float *data = (float*)PyArray_DATA(x);

    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_FLOAT32);
    float *result_data = (float*)PyArray_DATA(result);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (npy_intp i = 0; i < size; i++) {
        result_data[i] = gelu_f(data[i]);
    }

    return (PyObject*)result;
}

static PyObject* sigmoid_impl(PyObject* self, PyObject* args) {
    PyArrayObject *x;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x)) {
        return NULL;
    }

    if (PyArray_TYPE(x) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "sigmoid requires float32 arrays");
        return NULL;
    }

    npy_intp size = PyArray_SIZE(x);
    float *data = (float*)PyArray_DATA(x);

    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_FLOAT32);
    float *result_data = (float*)PyArray_DATA(result);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (npy_intp i = 0; i < size; i++) {
        result_data[i] = sigmoid_f(data[i]);
    }

    return (PyObject*)result;
}

static PyObject* tanh_impl(PyObject* self, PyObject* args) {
    PyArrayObject *x;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x)) {
        return NULL;
    }

    if (PyArray_TYPE(x) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "tanh requires float32 arrays");
        return NULL;
    }

    npy_intp size = PyArray_SIZE(x);
    float *data = (float*)PyArray_DATA(x);

    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_FLOAT32);
    float *result_data = (float*)PyArray_DATA(result);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (npy_intp i = 0; i < size; i++) {
        result_data[i] = tanh_f(data[i]);
    }

    return (PyObject*)result;
}

static PyObject* addmm_impl(PyObject* self, PyObject* args) {
    PyArrayObject *input, *mat1, *mat2, *bias;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &input, &PyArray_Type, &mat1, 
                          &PyArray_Type, &mat2, &PyArray_Type, &bias)) {
        return NULL;
    }

    if (PyArray_TYPE(input) != NPY_FLOAT32 || PyArray_TYPE(mat1) != NPY_FLOAT32 || 
        PyArray_TYPE(mat2) != NPY_FLOAT32 || PyArray_TYPE(bias) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "addmm requires float32 arrays");
        return NULL;
    }

    npy_intp *dims_input = PyArray_DIMS(input);
    npy_intp *dims_mat1 = PyArray_DIMS(mat1);
    npy_intp *dims_mat2 = PyArray_DIMS(mat2);
    
    int batch = (int)dims_input[0];
    int m = (int)dims_input[1];
    int n = (int)dims_mat2[1];

    float *data_input = (float*)PyArray_DATA(input);
    float *data_mat1 = (float*)PyArray_DATA(mat1);
    float *data_mat2 = (float*)PyArray_DATA(mat2);
    float *data_bias = (float*)PyArray_DATA(bias);

    npy_intp dims_out[2] = {batch, n};
    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(2, dims_out, NPY_FLOAT32);
    float *result_data = (float*)PyArray_DATA(result);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < n; j++) {
            float sum = data_bias[j];
            for (int p = 0; p < m; p++) {
                sum += data_input[b * m + p] * data_mat2[p * n + j];
            }
            result_data[b * n + j] = sum;
        }
    }

    return (PyObject*)result;
}

static PyObject* batch_norm_impl(PyObject* self, PyObject* args) {
    PyArrayObject *x, *weight, *bias, *running_mean, *running_var;
    float eps = 1e-5f;
    int training = 1;
    
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!|fi", &PyArray_Type, &x, &PyArray_Type, &weight, &PyArray_Type, &bias,
                          &PyArray_Type, &running_mean, &PyArray_Type, &running_var, &eps, &training)) {
        return NULL;
    }

    if (PyArray_TYPE(x) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "batch_norm requires float32 arrays");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(x);
    int batch = (int)dims[0];
    int channels = (int)dims[1];
    int spatial_size = 1;
    for (int i = 2; i < PyArray_NDIM(x); i++) {
        spatial_size *= (int)dims[i];
    }

    float *data_x = (float*)PyArray_DATA(x);
    float *data_w = (float*)PyArray_DATA(weight);
    float *data_b = (float*)PyArray_DATA(bias);
    float *data_mean = (float*)PyArray_DATA(running_mean);
    float *data_var = (float*)PyArray_DATA(running_var);

    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(x), dims, NPY_FLOAT32);
    float *result_data = (float*)PyArray_DATA(result);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int c = 0; c < channels; c++) {
        float mean, var;
        
        if (training) {
            float sum = 0.0f;
            for (int b = 0; b < batch; b++) {
                for (int s = 0; s < spatial_size; s++) {
                    sum += data_x[b * channels * spatial_size + c * spatial_size + s];
                }
            }
            mean = sum / (batch * spatial_size);
            
            float var_sum = 0.0f;
            for (int b = 0; b < batch; b++) {
                for (int s = 0; s < spatial_size; s++) {
                    float diff = data_x[b * channels * spatial_size + c * spatial_size + s] - mean;
                    var_sum += diff * diff;
                }
            }
            var = var_sum / (batch * spatial_size);
            
            data_mean[c] = 0.9f * data_mean[c] + 0.1f * mean;
            data_var[c] = 0.9f * data_var[c] + 0.1f * var;
        } else {
            mean = data_mean[c];
            var = data_var[c];
        }
        
        float std = sqrt(var + eps);
        
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < spatial_size; s++) {
                int idx = b * channels * spatial_size + c * spatial_size + s;
                result_data[idx] = (data_x[idx] - mean) / std * data_w[c] + data_b[c];
            }
        }
    }

    return (PyObject*)result;
}