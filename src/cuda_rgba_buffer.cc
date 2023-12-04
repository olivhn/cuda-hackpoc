#include "cuda_rgba_buffer.h"

#include <stdexcept>

#include <cuda_runtime.h>

namespace cuda_hackpoc::cuda
{

    std::unique_ptr<rgba_buffer> rgba_buffer::create(unsigned int width, unsigned int height)
    {
        size_t size = width * height * 4;

        void *cudaBuffer = nullptr;
        if (cudaMalloc(&cudaBuffer, size))
        {
            throw std::runtime_error("CUDA malloc.");
        }

        return std::unique_ptr<rgba_buffer>(new rgba_buffer(static_cast<unsigned char *>(cudaBuffer), width, height));
    }

    rgba_buffer::rgba_buffer(unsigned char *address, unsigned int width, unsigned int height)
        : address_(address), width_(width), height_(height){};

    rgba_buffer::~rgba_buffer()
    {
        cudaFree(this->address_);
    };

}
