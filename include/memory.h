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
    dptr* ptr;
    extent      size;
    mem_loc_t   loc;
    mem_algn_t  alignment;
    boolean     is_owner;
} mem_block;


extern mem_block* mem_alloc(extent size, mem_loc_t loc);
extern boolean    mem_free(mem_block* block); 
extern boolean    mem_copy(mem_block* dst, const mem_block* src, extent size);
extern boolean    mem_view(mem_block* dst, const dptr* data, mem_loc_t loc);
extern boolean    mem_view_copy(mem_block* dst, const dptr* data, mem_loc_t loc);

#endif /* MEMORY_H */