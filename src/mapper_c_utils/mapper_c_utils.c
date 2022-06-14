#include <Python.h>
#include <stdio.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

typedef struct vector_float32
{
    float x;
    float y;
    float z;
} vector_float32;

typedef struct vector_int32
{
    int32_t x;
    int32_t y;
    int32_t z;
} vector_int32;

// Divides vec_1 by vec_2.
inline void vector_divide(vector_float32 *vec_1, vector_float32 *vec_2)
{
    vec_1->x /= vec_2->x;
    vec_1->y /= vec_2->y;
    vec_1->z /= vec_2->z;
}

// Subtracts vec_2 from vec_1.
inline void vector_subtract(vector_float32 *vec_1, vector_float32 *vec_2)
{
    vec_1->x -= vec_2->x;
    vec_1->y -= vec_2->y;
    vec_1->z -= vec_2->z;
}

// Print the vector.
void vector_print(vector_float32 *vec)
{
    printf("x: %f, ", vec->x);
    printf("y: %f, ", vec->y);
    printf("z: %f\n", vec->z);
}

// As above, but for an int vector.
void vector_int_print(vector_int32 *vec)
{
    printf("x: %i, ", vec->x);
    printf("y: %i, ", vec->y);
    printf("z: %i\n", vec->z);
}

// Flattens the [x, y, z] element of a 3D array to a single index used to access
// the same element in an equivalent 1D array.
inline int offset(vector_int32 *index, vector_int32 *shape)
{
    return (index->x * shape->z * shape->y) + (index->y * shape->z) + index->z;
}

static PyObject *weighted_bin_3d(PyObject *dummy, PyObject *args)
{
    // Some object declarations.
    PyObject *coord_arg = NULL;
    PyObject *start_arg = NULL;
    PyObject *stop_arg = NULL;
    PyObject *step_arg = NULL;
    PyObject *shape_arg = NULL;
    PyObject *weights_arg = NULL;
    PyObject *out_arg = NULL;

    // Parse the arguments given to this function on the python end.
    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &coord_arg, &start_arg, &stop_arg, &step_arg,
                          &shape_arg, &weights_arg, &out_arg))
        return NULL;

    // Do some housework. We need to go from python objects to C types, which
    // requires some casting.
    PyObject *coords =
        PyArray_FROM_OTF(coord_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *start_arr =
        PyArray_FROM_OTF(start_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *stop_arr =
        PyArray_FROM_OTF(stop_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *step_arr =
        PyArray_FROM_OTF(step_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *shape_arr =
        PyArray_FROM_OTF(shape_arg, NPY_INT32, NPY_IN_ARRAY);
    PyObject *weights_arr =
        PyArray_FROM_OTF(weights_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *out_arr =
        PyArray_FROM_OTF(out_arg, NPY_FLOAT32, NPY_IN_ARRAY);

    float *coords_pointer = PyArray_GETPTR1(coords, 0);
    float *weights = PyArray_GETPTR1(weights_arr, 0);
    float *out = PyArray_GETPTR1(out_arr, 0);
    vector_float32 *start = (vector_float32 *)PyArray_GETPTR1(start_arr, 0);
    vector_float32 *stop = (vector_float32 *)PyArray_GETPTR1(stop_arr, 0);
    vector_float32 *step = (vector_float32 *)PyArray_GETPTR1(step_arr, 0);
    vector_int32 *shape = (vector_int32 *)PyArray_GETPTR1(shape_arr, 0);

    npy_intp *number_of_vectors = PyArray_SHAPE((PyArrayObject *)coords);
    // printf("Number of vectors found: %li\n", *number_of_vectors);
    // fflush(stdout);

    // This is where the heavy lifting takes place. This loop bottlenecks.
    for (int vector_num = 0; vector_num < *number_of_vectors; ++vector_num)
    {
        vector_float32 *current_coord =
            (vector_float32 *)(coords_pointer + vector_num * 3);
        // Deal with points being out of bounds.
        if (current_coord->x < start->x || current_coord->x >= stop->x)
            continue;
        if (current_coord->y < start->y || current_coord->y >= stop->y)
            continue;
        if (current_coord->z < start->z || current_coord->z >= stop->z)
            continue;
        // vector_print(current_coord);

        vector_subtract(current_coord, start);
        // vector_print(current_coord);

        vector_divide(current_coord, step);
        vector_int32 indices = {(int)current_coord->x,
                                (int)current_coord->y,
                                (int)current_coord->z};

        // printf("Indices calcuclated: ");
        // vector_int_print(&indices);

        // This point is within bounds. Add its weight to the weights array.
        int final_arr_idx = offset(&indices, shape);
        out[final_arr_idx] += weights[vector_num];
    }

    // Do some more housework: try not to leak memory.
    Py_DECREF(coords);
    Py_DECREF(start_arr);
    Py_DECREF(stop_arr);
    Py_DECREF(step_arr);
    Py_DECREF(shape_arr);
    Py_DECREF(weights_arr);
    Py_DECREF(out_arr);

    Py_IncRef(Py_None);
    return Py_None;
}

static PyObject *linear_map(PyObject *dummy, PyObject *args)
{
    // Pointers to the arguments we're going to receive (one matrix and one
    // array of vectors).
    PyObject *matrix_arg = NULL;
    PyObject *vector_array_arg = NULL;

    // Parse these arguments.
    if (!PyArg_ParseTuple(args, "OO", &vector_array_arg, &matrix_arg))
        return NULL;

    // Cast them to numpy arrays of floats.
    PyObject *matrix_arr = PyArray_FROM_OTF(matrix_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    float *matrix = PyArray_GETPTR1(matrix_arr, 0);
    PyObject *vector_array = PyArray_FROM_OTF(vector_array_arg,
                                              NPY_FLOAT32, NPY_IN_ARRAY);
    float *vector_array_pointer = PyArray_GETPTR1(vector_array, 0);

    // Work out how many vectors we're dealing with here.
    npy_intp *number_of_vectors = PyArray_SHAPE((PyArrayObject *)vector_array);

    // Iterate over each of the vectors and map them by the matrix.
    for (int i = 0; i < number_of_vectors; ++i)
    {
        vector_float32 *current_vector =
            (vector_float32 *)(vector_array_pointer + i * 3);

        // We're going to need to copy the current vector's elements.
        float x = current_vector->x;
        float y = current_vector->y;
        float z = current_vector->z;

        // Now update the current_vector via rules of matrix multiplication.
        current_vector->x = matrix[0] * x + matrix[1] * y + matrix[2] * z;
        current_vector->y = matrix[3] * x + matrix[4] * y + matrix[5] * z;
        current_vector->z = matrix[6] * x + matrix[7] * y + matrix[8] * z;
    }

    // Do the usual housework: don't leak memory and return None.
    Py_DECREF(matrix);
    Py_DECREF(vector_array);

    Py_IncRef(Py_None);
    return Py_None;
}

static PyMethodDef mapper_c_utils_methods[] = {
    {
        "weighted_bin_3d",
        weighted_bin_3d,
        METH_VARARGS,
        "Custom high performance weighted 3d binning tool.",
    },
    {
        "linear_map",
        linear_map,
        METH_VARARGS,
        "Custom high performance mapping of array of vectors by matrix.",
    },
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef mapper_c_utils_definition = {
    PyModuleDef_HEAD_INIT,
    "mapper_c_utils",
    "A Python module for highly optimized binning routines.",
    -1,
    mapper_c_utils_methods};

PyMODINIT_FUNC PyInit_mapper_c_utils(void)
{
    Py_Initialize();
    import_array();
    return PyModule_Create(&mapper_c_utils_definition);
}
