#include <iostream>
#include <stdexcept>
#include <chrono>

#include <GL\glew.h>

#include "cuda_rgba_buffer.h"
#include "cuda_texture_bridge.h"
#include "cuda_plasma.h"
#include "gl_texture_backed_framebuffer.h"
#include "sdl_gl_window.h"

using namespace cuda_hackpoc;

const char *kWindowTitle = "CUDA HackPOC";
const int kWindowWidth = 1600;
const int kWindowHeight = 1200;
const int kOpenGLMajorVersion = 3;
const int kOpenGLMinorVersion = 1;

void main_runloop(auto const &window, auto const &framebuffer, auto &cudaTextureBridge, auto const &cudaBuffer)
{
    auto runloopStartTime = std::chrono::steady_clock::now();

    SDL_Event e;
    while (true)
    {
        while (SDL_PollEvent(&e) != 0)
        {
            if (e.type == SDL_QUIT)
            {
                return;
            }
        }

        auto tick = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - runloopStartTime).count();

        cuda::render_plasma(cudaBuffer, tick);

        cudaTextureBridge.render_to_texture(cudaBuffer);

        window.render_fullscreen(framebuffer);
        window.swap_buffers();
    }
}

int main(int argc, char *args[])
{
    try
    {
        std::shared_ptr window = sdl::gl_window::create(kWindowTitle, kWindowWidth, kWindowHeight, kOpenGLMajorVersion, kOpenGLMinorVersion);

        auto glewInitResult = glewInit();
        if (glewInitResult != GLEW_OK)
        {
            // `glewGetErrorString` returns `unsigned char *`. 🤷
            throw std::runtime_error(reinterpret_cast<const char *>(glewGetErrorString(glewInitResult)));
        }

        std::shared_ptr framebuffer = gl::texture_backed_framebuffer::create(window, window->width(), window->height());
        auto cudaBuffer = cuda::rgba_buffer::create(framebuffer->width(), framebuffer->height());
        auto cudaTextureBridge = cuda::texture_bridge::create(framebuffer->shared_texture());

        main_runloop(*window, *framebuffer, *cudaTextureBridge, *cudaBuffer);
    }
    catch (const std::runtime_error &error)
    {
        std::cerr << "FATAL: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
