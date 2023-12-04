#ifndef cuda_hackpoc_cuda_texture_bridge_h
#define cuda_hackpoc_cuda_texture_bridge_h

#include <memory>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cuda_rgba_buffer.h"
#include "gl_texture.h"

namespace cuda_hackpoc::cuda
{

    class texture_bridge final
    {
    public:
        static std::unique_ptr<texture_bridge> create(std::shared_ptr<gl::texture> texture);

        void render_to_texture(cuda::rgba_buffer const &cudaBuffer);

        ~texture_bridge();

        texture_bridge(const texture_bridge &other) = delete;
        texture_bridge &operator=(const texture_bridge &other) = delete;

    private:
        texture_bridge(std::shared_ptr<gl::texture> && texture, cudaGraphicsResource *cudaTextureResource);

        std::shared_ptr<gl::texture> texture_;
        cudaGraphicsResource *cudaTextureResource_;
    };

}

#endif
