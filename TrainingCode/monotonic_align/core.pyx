
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def maximum_path_c(
    float[:, ::1] path, 
    float[:, ::1] value, 
    int t_y, 
    int t_x
):
    cdef int x, y
    cdef float v_prev, v_cur

    with nogil:
        # Loop over a single sample (Not the whole batch)
        for y in range(t_y):
            for x in range(t_x):
                if x == 0:
                    if y == 0:
                        path[y, x] = 1.0
                        continue
                    else:
                        path[y, x] = 0.0
                        continue

                v_cur = value[y, x]
                v_prev = value[y, x-1]

                if y > 0:
                    if v_prev < value[y-1, x-1]:
                        v_prev = value[y-1, x-1]

                if v_cur >= v_prev:
                    path[y, x] = 1.0
                else:
                    path[y, x] = 0.0
