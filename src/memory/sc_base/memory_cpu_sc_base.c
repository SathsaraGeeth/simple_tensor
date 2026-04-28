// memory_cpu.c - scalar cpu with 64B alignement
#include "memory.h"
#include <stdlib.h>
#include <string.h>

#define CACHE_LINE_SIZE 64

mem_block* mem_alloc(extent size, mem_loc_t loc) {
    mem_block* block = malloc(sizeof(mem_block));
    if (!block) return NULL;

    extent aligned_size = (size + CACHE_LINE_SIZE - 1) & ~(size_t)(CACHE_LINE_SIZE - 1);

    block->ptr       = aligned_alloc(CACHE_LINE_SIZE, aligned_size);
    block->size      = size;
    block->loc       = loc;
    block->alignment = CACHE_LINE;
    block->is_owner  = true;

    if (!block->ptr) {
        free(block);
        return NULL;
    }

    return block;
}

boolean mem_free(mem_block* block) {
    if (!block) return 1;

    if (block->is_owner && block->ptr) {
        free(block->ptr);
    }
    free(block);

    return 0;
}

boolean mem_copy(mem_block* dst, const mem_block* src, extent size) {
    if (!dst || !src)           return 1;
    if (!dst->ptr || !src->ptr) return 1;
    if (size > dst->size)       return 1;
    if (size > src->size)       return 1;

    memcpy(dst->ptr, src->ptr, size);
    return false;
}

boolean mem_view_copy(mem_block* dst, const dptr* data, mem_loc_t loc) {
    if (!dst || !data)  return true;
    if (!dst->ptr)      return true;
    if (!dst->size)      return true;

    memcpy(dst->ptr, data, dst->size);
    return false;
}

boolean mem_view(mem_block* dst, const dptr* data, mem_loc_t loc) {
    if (!dst || !data)  return true;
    if (!dst->size)      return true;

    if (dst->is_owner && dst->ptr) {
        free(dst->ptr);
    }

    dst->ptr = (dptr*)data;
    dst->is_owner = false;
    return false;
}
