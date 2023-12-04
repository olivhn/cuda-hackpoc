#include "gl_texture_backed_framebuffer.h"

#include <GL\glew.h>

namespace cuda_hackpoc::gl
{

    std::unique_ptr<texture_backed_framebuffer> texture_backed_framebuffer::create(std::shared_ptr<gl::context> context, GLuint width, GLuint height)
    {
        std::shared_ptr texture = gl::texture::create(context, width, height);
        auto textureId = texture->id();

        std::shared_ptr framebuffer = gl::framebuffer::create(context);
        auto frameBufferId = framebuffer->id();

        context->run_gl([&textureId, &frameBufferId]
                        {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferId);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureId, 0);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0); });

        return std::unique_ptr<texture_backed_framebuffer>(new texture_backed_framebuffer(std::move(context), std::move(texture), std::move(framebuffer)));
    }

    texture_backed_framebuffer::texture_backed_framebuffer(std::shared_ptr<gl::context> && context, std::shared_ptr<gl::texture> && texture, std::shared_ptr<gl::framebuffer> && framebuffer)
        : context_(std::move(context)), texture_(std::move(texture)), framebuffer_(std::move(framebuffer)){};

}
