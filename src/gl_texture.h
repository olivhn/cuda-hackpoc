#ifndef cuda_hackpoc_gl_texture_h
#define cuda_hackpoc_gl_texture_h

#include <memory>

#include <GL\glew.h>

#include "gl_context.h"

namespace cuda_hackpoc::gl
{

    class texture final
    {
    public:
        static std::unique_ptr<texture> create(std::shared_ptr<gl::context> context, GLuint width, GLuint height);
        static const GLint format = GL_RGBA;

        inline GLuint id() const { return this->id_; }
        inline int width() const { return this->width_; }
        inline int height() const { return this->height_; }
        
        inline gl::context const &context() const { return *this->context_; }

        ~texture();

        texture(const texture &other) = delete;
        texture &operator=(const texture &other) = delete;

    private:
        texture(std::shared_ptr<gl::context> && context, GLuint id, GLuint width, GLuint height);

        std::shared_ptr<gl::context> context_;
        GLuint id_;
        GLuint width_;
        GLuint height_;
    };

}

#endif
