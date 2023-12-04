#include "sdl_gl_window.h"

#include <cassert>
#include <stdexcept>

namespace cuda_hackpoc::sdl
{
    gl_window::unique_window_ptr gl_window::create_gl_window(const char *title, int width, int height)
    {
        auto window = SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
        if (window == nullptr)
        {
            throw std::runtime_error(SDL_GetError());
        }

        return gl_window::unique_window_ptr(window, SDL_DestroyWindow);
    }

    gl_window::unique_sdl_glcontext_ptr gl_window::create_sdl_glcontext(SDL_Window *window)
    {
        auto sdlGlContext = SDL_GL_CreateContext(window);
        if (sdlGlContext == nullptr)
        {
            throw std::runtime_error(SDL_GetError());
        }

        return gl_window::unique_sdl_glcontext_ptr(sdlGlContext, SDL_GL_DeleteContext);
    }

    std::unique_ptr<gl_window> gl_window::create(const char *title, int width, int height, int glMajor, int glMinor)
    {
        auto sdlInitGuard = sdl::init_guard::acquire();

        if (SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, glMajor) < 0)
        {
            throw std::runtime_error(SDL_GetError());
        }
        if (SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, glMinor) < 0)
        {
            throw std::runtime_error(SDL_GetError());
        }
        if (SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE) < 0)
        {
            throw std::runtime_error(SDL_GetError());
        }

        auto window = create_gl_window(title, width, height);
        auto sdlGlContext = create_sdl_glcontext(window.get());

        if (SDL_GL_SetSwapInterval(1) < 0)
        {
            throw std::runtime_error(SDL_GetError());
        }

        return std::unique_ptr<gl_window>(new gl_window(std::move(sdlInitGuard), std::move(window), std::move(sdlGlContext), width, height));
    }

    gl_window::gl_window(std::shared_ptr<sdl::init_guard> && initGuard, unique_window_ptr &&window, unique_sdl_glcontext_ptr &&sdlGlContext, int width, int height)
        : gl::context(), initGuard_(std::move(initGuard)), window_(std::move(window)), sdlGlContext_(std::move(sdlGlContext)), width_(width), height_(height)
    {
    }

    void gl_window::run_gl(std::function<void()> block) const
    {
        SDL_GL_MakeCurrent(this->window_.get(), this->sdlGlContext_.get());
        block();

        auto error = glGetError();
        if (error != GL_NO_ERROR)
        {
            // `gluErrorString` returns `unsigned char *`. 🤷
            throw std::runtime_error(reinterpret_cast<const char *>(gluErrorString(error)));
        }
    }

    void gl_window::render_fullscreen(gl::texture_backed_framebuffer const &fullscreenTexture) const
    {
        assert(dynamic_cast<gl_window const *>(&fullscreenTexture.context()) == this);

        this->run_gl([this, &fullscreenTexture]
                     {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, fullscreenTexture.framebuffer().id());
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glBlitFramebuffer(0, 0, fullscreenTexture.width(), fullscreenTexture.height(), 0, 0, this->width_, this->height_, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); });
    }

    void gl_window::swap_buffers() const
    {
        SDL_GL_SwapWindow(this->window_.get());
    }

}