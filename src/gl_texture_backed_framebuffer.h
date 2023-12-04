#ifndef cuda_hackpoc_gl_texture_backed_framebuffer_h
#define cuda_hackpoc_gl_texture_backed_framebuffer_h

#include <memory>

#include "gl_context.h"
#include "gl_framebuffer.h"
#include "gl_texture.h"

namespace cuda_hackpoc::gl
{

    class texture_backed_framebuffer final
    {
    public:
        static std::unique_ptr<texture_backed_framebuffer> create(std::shared_ptr<gl::context> context, GLuint width, GLuint height);

        inline int width() const { return this->texture_->width(); }
        inline int height() const { return this->texture_->height(); }

        inline gl::framebuffer const &framebuffer() const { return *this->framebuffer_; }
        inline gl::context const &context() const { return *this->context_; }

        inline std::shared_ptr<gl::texture> shared_texture() const { return this->texture_; }

        texture_backed_framebuffer(const texture_backed_framebuffer &other) = delete;
        texture_backed_framebuffer &operator=(const texture_backed_framebuffer &other) = delete;

    private:
        texture_backed_framebuffer(std::shared_ptr<gl::context> && context, std::shared_ptr<gl::texture> && texture, std::shared_ptr<gl::framebuffer> && framebuffer);

        std::shared_ptr<gl::context> context_;
        std::shared_ptr<gl::texture> texture_;
        std::shared_ptr<gl::framebuffer> framebuffer_;
    };

}

#endif