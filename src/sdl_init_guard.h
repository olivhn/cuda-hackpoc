#ifndef cuda_hackpoc_sdl_init_guard_h
#define cuda_hackpoc_sdl_init_guard_h

#include <memory>

namespace cuda_hackpoc::sdl
{
    class init_guard final
    {
    public:
        static std::shared_ptr<init_guard> acquire();

        ~init_guard();

        init_guard(const init_guard &other) = delete;
        init_guard &operator=(const init_guard &other) = delete;

    private:
        init_guard();
    };

}

#endif
