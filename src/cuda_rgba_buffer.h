#ifndef cuda_hackpoc_cuda_rgba_buffer_h
#define cuda_hackpoc_cuda_rgba_buffer_h

#include <memory>

namespace cuda_hackpoc::cuda
{

    class rgba_buffer final
    {
    public:
        static std::unique_ptr<rgba_buffer> create(unsigned int width, unsigned int height);

        inline unsigned char *address() const { return this->address_; }
        inline int width() const { return this->width_; }
        inline int height() const { return this->height_; }

        ~rgba_buffer();

        rgba_buffer(const rgba_buffer &other) = delete;
        rgba_buffer &operator=(const rgba_buffer &other) = delete;

    private:
        rgba_buffer(unsigned char *address, unsigned int width, unsigned int height);

        unsigned char *address_;
        unsigned int width_;
        unsigned int height_;
    };

}

#endif
