#ifndef DTYPE_H
#define DTYPE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef int8_t      int8;
typedef int16_t     int16;
typedef int32_t     int32;
typedef int64_t     int64;

typedef uint8_t     uint8;
typedef uint16_t    uint16;
typedef uint32_t    uint32;
typedef uint64_t    uint64;

typedef float       real32;
typedef double      real64;

typedef bool        boolean;
typedef size_t      extent;
typedef uint64_t    offset;

typedef void        dptr;

#endif /* DTYPE_H */
