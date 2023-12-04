#ifndef cuda_hackpoc_gl_framebuffer_h
#define cuda_hackpoc_gl_framebuffer_h

#include <memory>

#include <GL\glew.h>

#include "gl_context.h"

namespace cuda_hackpoc::gl
{

    class framebuffer final
    {
    public:
        static std::unique_ptr<framebuffer> create(std::shared_ptr<gl::context> context);

        inline GLuint id() const { return this->id_; }

        ~framebuffer();

        framebuffer(const framebuffer &other) = delete;
        framebuffer &operator=(const framebuffer &other) = delete;

    private:
        framebuffer(std::shared_ptr<gl::context> && context, GLuint id);

        std::shared_ptr<gl::context> context_;
        GLuint id_;
    };

}

#endif
