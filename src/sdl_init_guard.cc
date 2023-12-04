#include "sdl_init_guard.h"

#include <stdexcept>

#include <SDL.h>

namespace cuda_hackpoc::sdl
{
    static std::weak_ptr<init_guard> activeGuard;

    std::shared_ptr<init_guard> init_guard::acquire()
    {
        if (!activeGuard.expired())
        {
            return activeGuard.lock();
        }

        auto newGuard = std::shared_ptr<init_guard>(new init_guard());
        activeGuard = newGuard;
        return newGuard;
    }

    init_guard::init_guard()
    {
        if (SDL_Init(SDL_INIT_VIDEO) < 0)
        {
            throw std::runtime_error(SDL_GetError());
        }
    }

    init_guard::~init_guard()
    {
        SDL_Quit();
    }
}
