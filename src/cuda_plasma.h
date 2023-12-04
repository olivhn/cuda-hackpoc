#ifndef cuda_hackpoc_cuda_plasma_h
#define cuda_hackpoc_cuda_plasma_h

#include "cuda_rgba_buffer.h"

namespace cuda_hackpoc::cuda
{
    void render_plasma(cuda::rgba_buffer const &buffer, unsigned long long tick);
}

#endif