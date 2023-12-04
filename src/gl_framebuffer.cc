#include "gl_framebuffer.h"

namespace cuda_hackpoc::gl
{

    std::unique_ptr<framebuffer> framebuffer::create(std::shared_ptr<gl::context> context)
    {
        GLuint frameBufferId = 0;

        context->run_gl([&frameBufferId]
                        { glGenFramebuffers(1, &frameBufferId); });

        return std::unique_ptr<framebuffer>(new framebuffer(std::move(context), frameBufferId));
    }

    framebuffer::framebuffer(std::shared_ptr<gl::context> && context, GLuint id)
        : context_(std::move(context)), id_(id){};

    framebuffer::~framebuffer()
    {
        auto frameBufferId = this->id_;

        this->context_->run_gl([frameBufferId]
                               { glDeleteFramebuffers(1, &frameBufferId); });
    };

}
