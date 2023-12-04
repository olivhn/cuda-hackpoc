#include <GL\glew.h>

#include "cuda_texture_bridge.h"

#include <cassert>
#include <stdexcept>

#include "cuda_rgba_buffer.h"

namespace cuda_hackpoc::cuda
{

    std::unique_ptr<texture_bridge> texture_bridge::create(std::shared_ptr<gl::texture> texture)
    {
        cudaGraphicsResource *cudaTextureResource = nullptr;

        texture->context().run_gl([&texture, &cudaTextureResource]
                                  {
            if (cudaGraphicsGLRegisterImage(&cudaTextureResource, texture->id(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard)) {
                throw std::runtime_error("cudaGraphicsGLRegisterImage");
            } });

        return std::unique_ptr<texture_bridge>(new texture_bridge(std::move(texture), cudaTextureResource));
    }

    void texture_bridge::render_to_texture(cuda::rgba_buffer const &cudaBuffer)
    {
        static_assert(gl::texture::format == GL_RGBA);
        assert(cudaBuffer.width() <= this->texture_->width());
        assert(cudaBuffer.height() <= this->texture_->height());

        if (cudaGraphicsMapResources(1, &this->cudaTextureResource_, 0))
        {
            throw std::runtime_error("cudaGraphicsMapResources");
        }

        try {
            cudaArray *textureBuffer = nullptr;
            if (cudaGraphicsSubResourceGetMappedArray(&textureBuffer, this->cudaTextureResource_, 0, 0))
            {
                throw std::runtime_error("cudaGraphicsSubResourceGetMappedArray");
            }
            if (cudaMemcpy2DToArray(
                    textureBuffer,
                    0,
                    0,
                    cudaBuffer.address(),
                    cudaBuffer.width() * 4,
                    cudaBuffer.width() * 4,
                    cudaBuffer.height(),
                    cudaMemcpyDeviceToDevice))
            {
                throw std::runtime_error("cudaMemcpyToArray");
            }
        } catch (...) {
            cudaGraphicsUnmapResources(1, &this->cudaTextureResource_, 0);
            throw;
        }

        if (cudaGraphicsUnmapResources(1, &this->cudaTextureResource_, 0))
        {
            throw std::runtime_error("cudaGraphicsUnmapResources");
        }

    }

    texture_bridge::texture_bridge(std::shared_ptr<gl::texture> && texture, cudaGraphicsResource *cudaTextureResource)
        : texture_(std::move(texture)), cudaTextureResource_(cudaTextureResource)
    {
    }

    texture_bridge::~texture_bridge()
    {
        cudaGraphicsUnregisterResource(this->cudaTextureResource_);
    }

}