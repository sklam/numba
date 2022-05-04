import ctypes


lib = ctypes.CDLL('libextarray.so')


class ExtArrayHandle(ctypes.Structure):
    # opaque struct
    pass

ExtArrayHandlePtr = ctypes.POINTER(ExtArrayHandle)

# ExtArrayHandle* extarray_alloc(size_t nbytes)
lib.extarray_alloc.restype = ExtArrayHandlePtr
lib.extarray_alloc.argtypes = [ctypes.c_size_t]

# void extarray_free(ExtArrayHandle *hldr)
lib.extarray_free.restype = None
lib.extarray_free.argtypes = [ExtArrayHandlePtr]

# void* extarray_getpointer(ExtArrayHandle *hldr)
lib.extarray_getpointer.restype = ctypes.c_void_p
lib.extarray_getpointer.argtypes = [ExtArrayHandlePtr]

# void* extarray_make_meminfo(ExtArrayHandle *hldr)
lib.extarray_make_meminfo.restype = ctypes.c_void_p
lib.extarray_make_meminfo.argtypes = [ExtArrayHandlePtr]

# export API
alloc = lib.extarray_alloc
free = lib.extarray_free
getpointer = lib.extarray_getpointer
getnbytes = lib.extarray_getnbytes
make_meminfo = lib.extarray_make_meminfo
