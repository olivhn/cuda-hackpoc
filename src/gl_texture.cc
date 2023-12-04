#include "gl_texture.h"

namespace cuda_hackpoc::gl
{

    std::unique_ptr<texture> texture::create(std::shared_ptr<gl::context> context, GLuint width, GLuint height)
    {
        GLuint textureId = 0;

        context->run_gl([&textureId, &width, &height]
                        {
            glEnable(GL_TEXTURE_2D);
            glGenTextures(1, &textureId);
            glBindTexture(GL_TEXTURE_2D, textureId);
            glTexImage2D(GL_TEXTURE_2D, 0, texture::format, width, height, 0, texture::format, GL_UNSIGNED_BYTE, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0); });

        return std::unique_ptr<texture>(new texture(std::move(context), textureId, width, height));
    }

    texture::texture(std::shared_ptr<gl::context> && context, GLuint id, GLuint width, GLuint height)
        : context_(std::move(context)), id_(id), width_(width), height_(height){};

    texture::~texture()
    {
        auto textureId = this->id_;

        this->context_->run_gl([textureId]
                               { glDeleteTextures(1, &textureId); });
    };

}
