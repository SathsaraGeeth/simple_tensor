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

    if (!block->ptr) {
        free(block);
        return NULL;
    }

    return block;
}

int8 mem_free(mem_block* block) {
    if (!block) return 1;

    free(block->ptr);
    free(block);

    return 0;
}

int8 mem_copy(mem_block* dst, const mem_block* src, extent size) {
    if (!dst || !src)           return 1;
    if (!dst->ptr || !src->ptr) return 1;
    if (size > dst->size)       return 1;
    if (size > src->size)       return 1;

    memcpy(dst->ptr, src->ptr, size);
    return false;
}
