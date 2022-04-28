#include <stdlib.h>


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
