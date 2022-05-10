#include <stdlib.h>
#include <string.h>
#include "numba/core/runtime/nrt.h"
#include "numba/core/runtime/nrt_external.h"


typedef struct {
    void *buffer;
    size_t nbytes;
    size_t refct;
} ExtArrayHandle;

ExtArrayHandle* extarray_alloc(size_t nbytes) {
    ExtArrayHandle *hldr = malloc(nbytes);
    hldr->buffer = malloc(nbytes);
    memset(hldr->buffer, 0, nbytes);
    hldr->nbytes = nbytes;
    hldr->refct = 1;
    return hldr;
}

void extarray_acquire(ExtArrayHandle *hldr) { 
    hldr->refct += 1;
}


void extarray_free(ExtArrayHandle *hldr) { 
    // printf("%p free %zu\n", hldr, hldr->refct);
    hldr->refct -= 1;
    if (hldr->refct == 0) {
        free(hldr->buffer);
        free(hldr);
    }
}

void* extarray_getpointer(ExtArrayHandle *hldr) {
    return hldr->buffer;
}

size_t extarray_getnbytes(ExtArrayHandle *hldr) {
    return hldr->nbytes;
}

size_t extarray_getrefcount(ExtArrayHandle *hldr) {
    return hldr->refct;
}




/* The following is adapted from numba/core/runtime/nrt.c 

TODO: Numba NRT needs to expose NRT_MemInfo_new.
TODO: The MemInfo struct and the functions are expanded.
*/
struct MemInfo {
    size_t            refct;
    NRT_dtor_function dtor;
    void              *dtor_info;
    void              *data;
    size_t            size;    /* only used for NRT allocated memory */
    NRT_ExternalAllocator *external_allocator;
    ExtArrayHandle   *handle;
};

static
void ExtArray_NRT_MemInfo_init(NRT_MemInfo *mi,void *data, size_t size,
                      NRT_dtor_function dtor, void *dtor_info,
                      NRT_ExternalAllocator *external_allocator,
                      ExtArrayHandle *handle)
{
    mi->refct = 1;  /* starts with 1 refct */
    mi->dtor = dtor;
    mi->dtor_info = dtor_info;
    mi->data = data;
    mi->size = size;
    mi->external_allocator = external_allocator;
    mi->handle = handle;
    NRT_Debug(nrt_debug_print("NRT_MemInfo_init mi=%p external_allocator=%p\n", mi, external_allocator));
    /* Update stats */
    // TheMSys.atomic_inc(&TheMSys.stats_mi_alloc);  // missing
}

static
NRT_MemInfo *ExtArray_NRT_MemInfo_new(void *data, size_t size,
                             NRT_dtor_function dtor, void *dtor_info,
                             ExtArrayHandle *handle)
{
    NRT_MemInfo *mi = malloc(sizeof(NRT_MemInfo));
    NRT_Debug(nrt_debug_print("NRT_MemInfo_new mi=%p\n", mi));
    ExtArray_NRT_MemInfo_init(mi, data, size, dtor, dtor_info, NULL, handle);
    return mi;
}

static
void custom_dtor(void* ptr, size_t size, void* info) {
    extarray_free(info);
}

NRT_MemInfo* extarray_make_meminfo(ExtArrayHandle* handle) {
    void* dtor_info = handle;
    return ExtArray_NRT_MemInfo_new(handle->buffer, handle->nbytes, custom_dtor, dtor_info, handle);
}


ExtArrayHandle* extarray_meminfo_gethandle(NRT_MemInfo* mi) {
    return mi->handle;
}