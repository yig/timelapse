from scipy.linalg.lapack import dptsv

def ptsv( D, E, B ):
    assert len(D) == B.shape[0]
    assert len(E)+1 >= len(D)
    
    '''
    D = numpy.require( D, dtype = ctypes.c_double, requirements = [ 'F_CONTIGUOUS', 'WRITEABLE' ] )
    E = numpy.require( E, dtype = ctypes.c_double, requirements = [ 'F_CONTIGUOUS', 'WRITEABLE' ] )
    B = numpy.require( B, dtype = ctypes.c_double, requirements = [ 'F_CONTIGUOUS', 'WRITEABLE' ] )
    
    N = ctypes.c_int( len(D) )
    NRHS = ctypes.c_int( B.shape[1] )
    LDB = ctypes.c_int( max( 1, B.shape[0] ) )
    INFO = ctypes.c_int( 0 )
    
    _dptsv(
        ctypes.byref( N ),
        ctypes.byref( NRHS ),
        
        D.ctypes.data_as( ctypes.POINTER( ctypes.c_double ) ),
        E.ctypes.data_as( ctypes.POINTER( ctypes.c_double ) ),
        B.ctypes.data_as( ctypes.POINTER( ctypes.c_double ) ),
        
        ctypes.byref( LDB ),
        ctypes.byref( INFO )
        )
    
    if INFO.value < 0:
        raise RuntimeError( 'dptsv: The %d-th argument had an illegal value' % (-INFO.value) )
    elif INFO.value > 0:
        raise RuntimeError( 'dptsv: The leading minor of order %d is not positive definite; the solution has not been computed.' % INFO.value )
    
    return B
    '''
    
    _, _, X, info = dptsv( D, E, B, 1, 1, 1 )
    
    if info < 0:
        raise RuntimeError( 'dptsv: The %d-th argument had an illegal value' % (-info) )
    elif info > 0:
        raise RuntimeError( 'dptsv: The leading minor of order %d is not positive definite; the solution has not been computed.' % info )
    
    return X
