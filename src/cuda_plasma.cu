#include "cuda_plasma.h"

#include <cassert>

#include <cuda_runtime.h>

namespace cuda_hackpoc::cuda
{

  __device__ static void gradientColorValue(double positionPercentage, unsigned char &outRed, unsigned char &outGreen, unsigned char &outBlue)
  {
    assert(positionPercentage >= 0.0 && positionPercentage <= 1.0);

    const double colorSteps = 6.0;
    const double stepLength = 1.0 / colorSteps;

    if (positionPercentage <= stepLength)
    {
      outRed = unsigned char(positionPercentage * colorSteps * 255.0);
      outGreen = 255;
      outBlue = 0;
    }
    else if (positionPercentage <= 2 * stepLength)
    {
      outRed = 255;
      outGreen = unsigned char(255.0 - (positionPercentage - stepLength) * colorSteps * 255.0);
      outBlue = 0;
    }
    else if (positionPercentage <= 3 * stepLength)
    {
      outRed = 255;
      outGreen = 0;
      outBlue = unsigned char((positionPercentage - 2.0 * stepLength) * colorSteps * 255.0);
    }
    else if (positionPercentage <= 4 * stepLength)
    {
      outRed = unsigned char(255.0 - (positionPercentage - 3.0 * stepLength) * colorSteps * 255.0);
      outGreen = 0;
      outBlue = 255;
    }
    else if (positionPercentage <= 5 * stepLength)
    {
      outRed = 0;
      outGreen = unsigned char((positionPercentage - 4.0 * stepLength) * colorSteps * 255.0);
      outBlue = 255;
    }
    else
    {
      outRed = 0;
      outGreen = 255;
      outBlue = unsigned char(255.0 - (positionPercentage - 5.0 * stepLength) * colorSteps * 255.0);
    }
  }

  __global__ static void plasma_kernel(unsigned char *cudaBuffer, int width, int height, unsigned long long tick)
  {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (!(x < width))
    {
      return;
    }

    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (!(y < height))
    {
      return;
    }

    const int bufferIndex = (y * width + x) * 4;

    const double normalizedX = (double(x) / double(width)) * 640.0;
    const double normalizedY = (double(y) / double(height)) * 480.0;

    double z = sin(hypot(200.0 - normalizedY, 320.0 - normalizedX) / 16.0 + tick / 300.0);
    z += sin(normalizedX / (37 + 15 * cos(normalizedY / 74))) * cos(normalizedY / (31 + 11 * sin(normalizedX / 57)) + tick / 250.0);
    z /= 2.0;
    z /= 2.0;
    z += 0.5;
    z += double(tick) / 1500.0;
    z -= floor(z);

    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;
    gradientColorValue(z, r, g, b);

    cudaBuffer[bufferIndex] = r;
    cudaBuffer[bufferIndex + 1] = g;
    cudaBuffer[bufferIndex + 2] = b;
    cudaBuffer[bufferIndex + 3] = 0xff;
  }

  void render_plasma(rgba_buffer const &buffer, unsigned long long tick)
  {
    dim3 block(16, 16, 1);

    int gridWidth = buffer.width() / block.x + (buffer.width() % block.x != 0 ? 1 : 0);
    int gridHeight = buffer.height() / block.y + (buffer.height() % block.y != 0 ? 1 : 0);
    dim3 grid(gridWidth, gridHeight, 1);

    plasma_kernel<<<grid, block>>>(buffer.address(), buffer.width(), buffer.height(), tick);

    cudaDeviceSynchronize();
  }

}