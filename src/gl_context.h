#ifndef cuda_hackpoc_gl_context_h
#define cuda_hackpoc_gl_context_h

#include <functional>

namespace cuda_hackpoc::gl
{
    class context
    {
    public:
        virtual void run_gl(std::function<void()> block) const = 0;

        context() = default;
        virtual ~context() {}
        context(const context &other) = delete;
        context &operator=(const context &other) = delete;
    };
}

#endif