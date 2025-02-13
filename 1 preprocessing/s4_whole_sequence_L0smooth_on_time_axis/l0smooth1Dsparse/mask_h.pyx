# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

ctypedef double DTYPE_t

cpdef mask_h_threechannel( DTYPE_t[:,:] h, DTYPE_t threshold, DTYPE_t[:] maskscratch ):
    assert h.shape[1] % 3 == 0
    
    cdef int row, col
    cdef int row_max = h.shape[0]
    cdef int col_max = h.shape[1]
    cdef DTYPE_t r
    
    ## Assume h is a contiguous column-major (Fortran) array
    cdef DTYPE_t* rawh = &h[0,0]
    cdef DTYPE_t *rawh0
    cdef DTYPE_t *rawh1
    cdef DTYPE_t *rawh2
    ## Assume maskscratch is a contiguous 1D array
    cdef DTYPE_t* rawmask0 = &maskscratch[0]
    cdef DTYPE_t* rawmask
    
    for col in range( 0, col_max, 3 ):
        
        rawmask = rawmask0
        for row in range( row_max ):
            # r = h[ <unsigned int>row, <unsigned int>col ]
            # rawmask[row] = r*r
            r = rawh[0]
            rawmask[0] = r*r
            rawh += 1
            rawmask += 1
        
        rawmask = rawmask0
        for row in range( row_max ):
            # r = h[ <unsigned int>row, <unsigned int>(col+1) ]
            # rawmask[row] += r*r
            r = rawh[0]
            rawmask[0] += r*r
            rawh += 1
            rawmask += 1
        
        rawmask = rawmask0
        for row in range( row_max ):
            # r = h[ <unsigned int>row, <unsigned int>(col+2) ]
            # rawmask[row] += r*r
            r = rawh[0]
            rawmask[0] += r*r
            rawh += 1
            rawmask += 1
        
        rawmask = rawmask0
        rawh2 = rawh - row_max
        rawh1 = rawh2 - row_max
        rawh0 = rawh1 - row_max
        for row in range( row_max ):
            if rawmask[0] < threshold:
                rawh0[0] = 0
                rawh1[0] = 0
                rawh2[0] = 0
            rawh0 += 1
            rawh1 += 1
            rawh2 += 1
            rawmask += 1
