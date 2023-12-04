#ifndef cuda_hackpoc_sdl_gl_window_h
#define cuda_hackpoc_sdl_gl_window_h

#include <functional>
#include <memory>

#include <GL\glew.h>
#include <SDL.h>
#include <SDL_opengl.h>

#include "gl_context.h"
#include "gl_texture_backed_framebuffer.h"
#include "sdl_init_guard.h"

namespace cuda_hackpoc::sdl
{

    class gl_window final : public gl::context
    {
    public:
        static std::unique_ptr<gl_window> create(const char *title, int width, int height, int glMajor, int glMinor);

        inline int width() const { return this->width_; }
        inline int height() const { return this->height_; }

        void run_gl(std::function<void()> block) const override;

        void render_fullscreen(gl::texture_backed_framebuffer const &fullscreenTexture) const;
        void swap_buffers() const;

        gl_window(const gl_window &other) = delete;
        gl_window &operator=(const gl_window &other) = delete;

    private:
        using unique_window_ptr = std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)>;
        using unique_sdl_glcontext_ptr = std::unique_ptr<void, decltype(&SDL_GL_DeleteContext)>;

        static unique_window_ptr create_gl_window(const char *title, int width, int height);
        static unique_sdl_glcontext_ptr create_sdl_glcontext(SDL_Window *window);

        gl_window(std::shared_ptr<sdl::init_guard> && initGuard, unique_window_ptr &&window, unique_sdl_glcontext_ptr &&sdlGlContext, int width, int height);

        // These have to be destructed in the correct order.
        std::shared_ptr<sdl::init_guard> initGuard_;
        unique_window_ptr window_;
        unique_sdl_glcontext_ptr sdlGlContext_;

        int width_;
        int height_;
    };

}

#endif
