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
inline uint64_t offset(vector_int32 *index, vector_int32 *shape)
{
    return (index->x * shape->z * shape->y) + (index->y * shape->z) + index->z;
}

static PyObject *weighted_bin_3d(PyObject *dummy, PyObject *args)
{
    // Some object declarations.
    PyObject *coord_arg = NULL;
    PyObject *start_arg = NULL;
    PyObject *step_arg = NULL;
    PyObject *shape_arg = NULL;
    PyObject *weights_arg = NULL;
    PyObject *out_arg = NULL;
    PyObject *count_arg = NULL;
    PyObject *min_intensity_arg = NULL;

    // Parse the arguments given to this function on the python end.
    if (!PyArg_ParseTuple(args, "OOOOOOOO",
                          &coord_arg, &start_arg, &step_arg,
                          &shape_arg, &weights_arg, &out_arg, &count_arg,
                          &min_intensity_arg))
        return NULL;
    // Do some housework. We need to go from python objects to C types, which
    // requires some casting.
    PyObject *coords =
        PyArray_FROM_OTF(coord_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *start_arr =
        PyArray_FROM_OTF(start_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *step_arr =
        PyArray_FROM_OTF(step_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *shape_arr =
        PyArray_FROM_OTF(shape_arg, NPY_INT32, NPY_IN_ARRAY);
    PyObject *weights_arr =
        PyArray_FROM_OTF(weights_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *out_arr =
        PyArray_FROM_OTF(out_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *count_arr =
        PyArray_FROM_OTF(count_arg, NPY_UINT32, NPY_IN_ARRAY);
    PyObject *min_intensity_arr =
        PyArray_FROM_OTF(min_intensity_arg, NPY_FLOAT32, NPY_IN_ARRAY);

    float min_intensity = *((float *)PyArray_GETPTR1(min_intensity_arr, 0));
    float *coords_pointer = PyArray_GETPTR1(coords, 0);
    float *weights = PyArray_GETPTR1(weights_arr, 0);
    float *out = PyArray_GETPTR1(out_arr, 0);
    uint32_t *count = PyArray_GETPTR1(count_arr, 0);

    vector_float32 *start = (vector_float32 *)PyArray_GETPTR1(start_arr, 0);
    vector_float32 *step = (vector_float32 *)PyArray_GETPTR1(step_arr, 0);
    vector_int32 *shape = (vector_int32 *)PyArray_GETPTR1(shape_arr, 0);

    npy_intp *coord_shape = PyArray_SHAPE((PyArrayObject *)coords);
    int number_of_vectors = coord_shape[0];

    // This is where the heavy lifting takes place. This loop bottlenecks.
    for (int vector_num = 0; vector_num < number_of_vectors; ++vector_num)
    {
        // Skip if this pixel is being masked based on intensity.
        if (weights[vector_num] < min_intensity)
            continue;

        // Skip if this pixel is being masked (signaled by NaN).
        // Note that NaN==NaN should give False, so we must use isnan.
        if (npy_isnan(weights[vector_num]))
            continue;

        vector_float32 *current_coord =
            (vector_float32 *)(coords_pointer + vector_num * 3);

        // Deal with points being below the lower bound.
        if (current_coord->x < start->x)
            continue;
        if (current_coord->y < start->y)
            continue;
        if (current_coord->z < start->z)
            continue;

        vector_subtract(current_coord, start);

        vector_divide(current_coord, step);
        vector_int32 indices = {(int)current_coord->x,
                                (int)current_coord->y,
                                (int)current_coord->z};

        // Deal with points being over the upper bound.
        // There are important, tedious floating point precision reasons why
        // the bounds checking must be done in two parts. Don't ask.
        if (indices.x >= shape->x)
            continue;
        if (indices.y >= shape->y)
            continue;
        if (indices.z >= shape->z)
            continue;

        // This point is within bounds. Add its weight to the weights array.
        uint64_t final_arr_idx = offset(&indices, shape);
        out[final_arr_idx] += weights[vector_num];
        count[final_arr_idx] += 1;
    }

    // Do some more housework: try not to leak memory.
    Py_DECREF(coords);
    Py_DECREF(start_arr);
    Py_DECREF(step_arr);
    Py_DECREF(shape_arr);
    Py_DECREF(weights_arr);
    Py_DECREF(out_arr);
    Py_DECREF(count_arr);
    Py_DECREF(min_intensity_arr);

    Py_IncRef(Py_None);
    return Py_None;
}

static PyObject *weighted_bin_1d(PyObject *dummy, PyObject *args)
{
    // Some object declarations.
    PyObject *coord_arg = NULL;
    float start;
    float step;
    float shape;
    PyObject *weights_arg = NULL;
    PyObject *out_arg = NULL;
    PyObject *count_arg = NULL;

    // Parse the arguments given to this function on the python end.
    if (!PyArg_ParseTuple(args, "OfffOOO",
                          &coord_arg, &start, &step, &shape,
                          &weights_arg, &out_arg, &count_arg))
        return NULL;

    // Do some housework. We need to go from python objects to C types, which
    // requires some casting.
    PyObject *coords =
        PyArray_FROM_OTF(coord_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *weights_arr =
        PyArray_FROM_OTF(weights_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *out_arr =
        PyArray_FROM_OTF(out_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *count_arr =
        PyArray_FROM_OTF(count_arg, NPY_UINT32, NPY_IN_ARRAY);

    float *coords_pointer = PyArray_GETPTR1(coords, 0);
    float *weights = PyArray_GETPTR1(weights_arr, 0);
    float *out = PyArray_GETPTR1(out_arr, 0);
    uint32_t *count = PyArray_GETPTR1(count_arr, 0);

    npy_intp *coord_shape = PyArray_SHAPE((PyArrayObject *)coords);
    int num_coords = coord_shape[0];

    // This is where the heavy lifting takes place. This loop bottlenecks.
    for (int i = 0; i < num_coords; ++i)
    {
        float current_coord = coords_pointer[i];

        // Deal with points being below the lower bound.
        if (current_coord < start)
            continue;

        current_coord -= start;
        current_coord /= step;
        int32_t index = (int)current_coord;

        // Deal with points being over the upper bound.
        // There are important, tedious floating point precision reasons why
        // the bounds checking must be done in two parts. Don't ask.
        if (index >= shape)
            continue;

        // This point is within bounds. Add its weight to the weights array.
        out[index] += weights[i];
        count[index] += 1;
    }

    // Do some more housework: try not to leak memory.
    Py_DECREF(coords);
    Py_DECREF(weights_arr);
    Py_DECREF(out_arr);
    Py_DECREF(count_arr);

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
    PyObject *matrix_arr =
        PyArray_FROM_OTF(matrix_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    float *matrix = PyArray_GETPTR1(matrix_arr, 0);
    PyObject *vector_array = PyArray_FROM_OTF(vector_array_arg,
                                              NPY_FLOAT32, NPY_IN_ARRAY);
    float *vector_array_pointer = PyArray_GETPTR1(vector_array, 0);

    // Work out how many vectors we're dealing with here.
    npy_intp *shape = PyArray_SHAPE((PyArrayObject *)vector_array);
    int number_of_vectors = shape[0];

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

    // Do the usual housework: don't leak memory; return None.
    Py_DECREF(matrix_arr);
    Py_DECREF(vector_array);

    Py_IncRef(Py_None);
    return Py_None;
}

static PyObject *lorentz_correction(PyObject *dummy, PyObject *args)
{
    PyObject *k_in_arg;
    PyObject *k_out_arg;
    PyObject *intensities_arg;

    if (!PyArg_ParseTuple(args, "OOO",
                          &k_in_arg, &k_out_arg, &intensities_arg))
        return NULL;

    // Convert to arrays.
    PyObject *k_in_arr =
        PyArray_FROM_OTF(k_in_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *k_out_arr =
        PyArray_FROM_OTF(k_out_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *intensities_arr =
        PyArray_FROM_OTF(intensities_arg, NPY_FLOAT32, NPY_IN_ARRAY);

    // Grab pointers.
    vector_float32 *k_in = PyArray_GETPTR1(k_in_arr, 0);
    float *k_out_ptr = PyArray_GETPTR1(k_out_arr, 0);
    float *intensities = PyArray_GETPTR1(intensities_arr, 0);

    // Work out how many vectors we're dealing with here.
    npy_intp *shape = PyArray_SHAPE((PyArrayObject *)k_out_arr);
    int number_of_vectors = shape[0];

    // Iterate over every vector and apply the correction.
    for (int i = 0; i < number_of_vectors; ++i)
    {
        // Note that this *** MUST ALREADY BE NORMALISED ***.
        vector_float32 *k_out = (vector_float32 *)(k_out_ptr + i * 3);

        // Work out sin^2(theta), where theta is half of the total angle through
        // which light has diffracted.
        float sin_sq_theta = 0.5F * (1.F - (k_out->x * k_in->x +
                                            k_out->y * k_in->y +
                                            k_out->z * k_in->z));
        float cos_theta = sqrt(1 - sin_sq_theta);

        // Apply the normalisation.
        intensities[i] *= cos_theta * sin_sq_theta;
    }

    // Tidy up; return None.
    Py_DECREF(k_in_arr);
    Py_DECREF(k_out_arr);
    Py_DECREF(intensities_arr);

    Py_IncRef(Py_None);
    return Py_None;
}

static PyObject *linear_pol_correction(PyObject *dummy, PyObject *args)
{
    PyObject *polarisation_vector_arg;
    PyObject *vector_array_arg;
    PyObject *intensities_arg;

    // Parse these arguments.
    if (!PyArg_ParseTuple(args, "OOO",
                          &polarisation_vector_arg, &vector_array_arg,
                          &intensities_arg))
        return NULL;

    // Convert to arrays.
    PyObject *polarisation_vector_arr =
        PyArray_FROM_OTF(polarisation_vector_arg, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *vector_array = PyArray_FROM_OTF(vector_array_arg,
                                              NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *intensities_arr = PyArray_FROM_OTF(intensities_arg,
                                                 NPY_FLOAT32, NPY_IN_ARRAY);

    // Grab pointers.
    vector_float32 *polarisation =
        PyArray_GETPTR1(polarisation_vector_arr, 0);
    float *vector_array_ptr = PyArray_GETPTR1(vector_array, 0);
    float *intensities = PyArray_GETPTR1(intensities_arr, 0);

    // Work out how many vectors we're dealing with here.
    npy_intp *shape = PyArray_SHAPE((PyArrayObject *)vector_array);
    int number_of_vectors = shape[0];

    // Iterate over every vector and apply the correction.
    for (int i = 0; i < number_of_vectors; ++i)
    {
        vector_float32 *k_out = (vector_float32 *)(vector_array_ptr + i * 3);

        // Carry out the dot product. Since both vectors are normalised, this
        // just gives us the cosine of the angle between them.
        float cos_phi = k_out->x * polarisation->x +
                        k_out->y * polarisation->y +
                        k_out->z * polarisation->z;

        // The polarisation correction is proportional to the square of the sine
        // of this angle.
        float sin_sq_phi = 1 - cos_phi * cos_phi;

        // Normalise the intensities.
        intensities[i] /= sin_sq_phi;
    }

    // Clean up the arrays that were made here.
    Py_DECREF(polarisation_vector_arr);
    Py_DECREF(vector_array);
    Py_DECREF(intensities_arr);

    // Inc and return None.
    Py_IncRef(Py_None);
    return Py_None;
}

static PyObject *cylindrical_polar(PyObject *dummy, PyObject *args)
{
    // Pointers to the arguments we're going to receive (one matrix and one
    // array of vectors).
    PyObject *vector_array_arg = NULL;

    // Parse these arguments.
    if (!PyArg_ParseTuple(args, "O", &vector_array_arg))
        return NULL;

    // Cast them to numpy arrays of floats.
    PyObject *vector_array = PyArray_FROM_OTF(vector_array_arg,
                                              NPY_FLOAT32, NPY_IN_ARRAY);
    float *vector_array_pointer = PyArray_GETPTR1(vector_array, 0);

    // Work out how many vectors we're dealing with here.
    npy_intp *shape = PyArray_SHAPE((PyArrayObject *)vector_array);
    int number_of_vectors = shape[0];

    // Iterate over each of the vectors and map them by the matrix.
    for (int i = 0; i < number_of_vectors; ++i)
    {
        vector_float32 *current_vector =
            (vector_float32 *)(vector_array_pointer + i * 3);

        // Make this code a bit less ugly.
        float x = current_vector->x;
        float y = current_vector->y;

        // Calculate the polar angle from the x-axis.
        float angle = atan2f(x, y);

        // Calculate the radius.
        float radius = sqrt(x * x + y * y);

        // Now update the current_vector to be in polar coords.
        current_vector->x = radius;
        current_vector->y = angle;
        // Note that current_vector->z is already correct in cylindrical polars.
    }

    // Do the usual housework: don't leak memory and return None.
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
        "weighted_bin_1d",
        weighted_bin_1d,
        METH_VARARGS,
        "Custom high performance weighted 1d binning tool.",
    },
    {
        "linear_pol_correction",
        linear_pol_correction,
        METH_VARARGS,
        ("Carries out an exact polarisation correction for arbitrarily "
         "linearly polarised light."),
    },
    {
        "lorentz_correction",
        lorentz_correction,
        METH_VARARGS,
        "Carries out an exact Lorentz correction.",
    },
    {
        "cylindrical_polar",
        cylindrical_polar,
        METH_VARARGS,
        "Maps input Nx3 vector to cylindrical polars (in degrees).",
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
