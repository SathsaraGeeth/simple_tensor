#ifndef MEMORY_H
#define MEMORY_H

#include "dtype.h"

typedef enum {
    MEM_HOST,
    MEM_DEVICE
} mem_loc_t;

typedef enum {
    CACHE_LINE,
    UNALIGNED
} mem_algn_t;

typedef struct {
    dptr*       ptr;
    extent      size;
    mem_loc_t   loc;
    mem_algn_t  alignment;
} mem_block;

extern mem_block* mem_alloc(extent size, mem_loc_t loc);
extern int8       mem_free(mem_block* block);                                        // free the mem_block
extern int8       mem_copy(mem_block* dst, const mem_block* src, extent size);       // copy the dst -> src

#endif /* MEMORY_H */