# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "helper.hxx" namespace "andres":
    cdef enum CoordinateOrder:
        FirstMajorOrder
        LastMajorOrder

    cdef cppclass npView[T]:
        npView(
            np.intp_t* dimensions_start,
            np.intp_t* dimensions_end,
            np.intp_t* strides_start,
            T* data,
            CoordinateOrder order
        )

cdef extern from "mc-infer.hxx":
    void solve(npView[np.float64_t], npView[np.float64_t], npView[np.float64_t], int, npView[np.int32_t])


def infer(np.ndarray[np.float64_t, ndim=2, mode="c"] unaries, np.ndarray[np.float64_t, ndim=2, mode="c"] general_pw, np.ndarray[np.float64_t, ndim=2, mode="c"] specific_pw, number_of_bg_classes, np.ndarray[np.int32_t, ndim=2, mode="c"] solution):
    '''
    '''
    cdef npView[np.float64_t]* unaries_view
    cdef np.ndarray[np.intp_t, ndim=1] unaries_strides
    unaries_strides = np.array([unaries.strides[j] for j in range(unaries.ndim)]).astype(np.intp) // unaries.itemsize
    unaries_view = new npView[np.float64_t](
      unaries.shape,
      unaries.shape + <int> unaries.ndim,
      <np.intp_t*> unaries_strides.data,
      <np.float64_t*> unaries.data,
      LastMajorOrder if unaries.flags['F_CONTIGUOUS'] else FirstMajorOrder
    )

    cdef npView[np.float64_t]* general_pw_view
    cdef np.ndarray[np.intp_t, ndim=1] general_pw_strides
    general_pw_strides = np.array([general_pw.strides[j] for j in range(general_pw.ndim)]).astype(np.intp) // general_pw.itemsize
    general_pw_view = new npView[np.float64_t](
      general_pw.shape,
      general_pw.shape + <int> general_pw.ndim,
      <np.intp_t*> general_pw_strides.data,
      <np.float64_t*> general_pw.data,
      LastMajorOrder if general_pw.flags['F_CONTIGUOUS'] else FirstMajorOrder
    )

    cdef npView[np.float64_t]* specific_pw_view
    cdef np.ndarray[np.intp_t, ndim=1] specific_pw_strides
    specific_pw_strides = np.array([specific_pw.strides[j] for j in range(specific_pw.ndim)]).astype(np.intp) // specific_pw.itemsize
    specific_pw_view = new npView[np.float64_t](
      specific_pw.shape,
      specific_pw.shape + <int> specific_pw.ndim,
      <np.intp_t*> specific_pw_strides.data,
      <np.float64_t*> specific_pw.data,
      LastMajorOrder if specific_pw.flags['F_CONTIGUOUS'] else FirstMajorOrder
    )

    cdef npView[np.int32_t]* solution_view
    cdef np.ndarray[np.intp_t, ndim=1] solution_strides
    solution_strides = np.array([solution.strides[j] for j in range(solution.ndim)]).astype(np.intp) // solution.itemsize
    solution_view = new npView[np.int32_t](
      solution.shape,
      solution.shape + <int> solution.ndim,
      <np.intp_t*> solution_strides.data,
      <np.int32_t*> solution.data,
      LastMajorOrder if solution.flags['F_CONTIGUOUS'] else FirstMajorOrder
    )

    solve(cython.operator.dereference(unaries_view), cython.operator.dereference(general_pw_view), cython.operator.dereference(specific_pw_view), number_of_bg_classes, cython.operator.dereference(solution_view))

    del unaries_view
    del general_pw_view
    del specific_pw_view
    del solution_view
