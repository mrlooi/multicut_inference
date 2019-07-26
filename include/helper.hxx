// helper.cpp - wrap templated functions in classes for cython to call

#include <numpy/arrayobject.h>
#include <andres/marray.hxx>


namespace andres {

template<class T>
class npView : public andres::View<T>
{
public:
    npView(npy_intp *shape_start, npy_intp *shape_end, npy_intp *strides_start, T *data, andres::CoordinateOrder order)
        : andres::View<T>(shape_start, shape_end, strides_start, data, order)
    {

    }
};

}
