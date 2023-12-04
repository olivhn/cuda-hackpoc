cmake_minimum_required(VERSION 3.10)

find_library(GLEW_LIB "glew32.lib" REQUIRED)
find_library(GLU_LIB "glu32.lib" REQUIRED)
find_library(OPENGL_LIB "opengl32.lib" REQUIRED)
find_library(SDL_LIB "SDL2.lib" REQUIRED)
find_library(SDL_MAIN_LIB "SDL2main.lib" REQUIRED)

find_path(GLEW_INCLUDE "GL\\glew.h" REQUIRED)
find_path(SDL_INCLUDE "SDL.h" REQUIRED)

project(cuda-hackpoc LANGUAGES CXX CUDA)

add_executable(cuda-hackpoc)
set_property(TARGET cuda-hackpoc PROPERTY CXX_STANDARD 20)
set_target_properties(cuda-hackpoc PROPERTIES CUDA_ARCHITECTURES native)
target_sources(cuda-hackpoc PRIVATE
    src/cuda_plasma.cu 
    src/cuda_rgba_buffer.cc 
    src/cuda_texture_bridge.cc 
    src/gl_framebuffer.cc
    src/gl_texture.cc 
    src/gl_texture_backed_framebuffer.cc 
    src/main.cc 
    src/sdl_gl_window.cc 
    src/sdl_init_guard.cc 
)
target_link_libraries(cuda-hackpoc PRIVATE 
    ${GLEW_LIB}
    ${GLU_LIB}
    ${OPENGL_LIB}
    ${SDL_LIB}
    ${SDL_MAIN_LIB}
)
target_include_directories(cuda-hackpoc PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${GLEW_INCLUDE}
    ${SDL_INCLUDE}
)
