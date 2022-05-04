#include <stdlib.h>
#include "numba/core/runtime/nrt.h"
#include "numba/core/runtime/nrt_external.h"


typedef struct {
    void *buffer;
    size_t nbytes;
} ExtArrayHandle;

ExtArrayHandle* extarray_alloc(size_t nbytes) {
    ExtArrayHandle *hldr = malloc(nbytes);
    hldr->buffer = malloc(nbytes);
    hldr->nbytes = nbytes;
    return hldr;
}

void extarray_free(ExtArrayHandle *hldr) { 
    free(hldr->buffer);
    free(hldr);
}

void* extarray_getpointer(ExtArrayHandle *hldr) {
    return hldr->buffer;
}

size_t extarray_getnbytes(ExtArrayHandle *hldr) {
    return hldr->nbytes;
}


/* The following is copied from numba/core/runtime/nrt.c 

TODO: Numba NRT needs to expose NRT_MemInfo_new
*/
struct MemInfo {
    size_t            refct;
    NRT_dtor_function dtor;
    void              *dtor_info;
    void              *data;
    size_t            size;    /* only used for NRT allocated memory */
    NRT_ExternalAllocator *external_allocator;
};

void NRT_MemInfo_init(NRT_MemInfo *mi,void *data, size_t size,
                      NRT_dtor_function dtor, void *dtor_info,
                      NRT_ExternalAllocator *external_allocator)
{
    mi->refct = 1;  /* starts with 1 refct */
    mi->dtor = dtor;
    mi->dtor_info = dtor_info;
    mi->data = data;
    mi->size = size;
    mi->external_allocator = external_allocator;
    NRT_Debug(nrt_debug_print("NRT_MemInfo_init mi=%p external_allocator=%p\n", mi, external_allocator));
    /* Update stats */
    // TheMSys.atomic_inc(&TheMSys.stats_mi_alloc);  // missing
}

NRT_MemInfo *NRT_MemInfo_new(void *data, size_t size,
                             NRT_dtor_function dtor, void *dtor_info)
{
    NRT_MemInfo *mi = malloc(sizeof(NRT_MemInfo));
    NRT_Debug(nrt_debug_print("NRT_MemInfo_new mi=%p\n", mi));
    NRT_MemInfo_init(mi, data, size, dtor, dtor_info, NULL);
    return mi;
}

static
void custom_dtor(void* ptr, size_t size, void* info) {
    extarray_free(info);
}

NRT_MemInfo* extarray_make_meminfo(ExtArrayHandle* handle) {
    void* dtor_info = handle;
    return NRT_MemInfo_new(handle->buffer, handle->nbytes, custom_dtor, dtor_info);
}
