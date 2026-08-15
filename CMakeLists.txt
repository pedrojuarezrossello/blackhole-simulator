cmake_minimum_required(VERSION 3.24)

project(kerr_black_hole_sfml LANGUAGES CXX)

# The simulation uses std::jthread, concepts and other C++20 facilities.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
# We explicitly find OpenGL rather than relying on SFML to pull it in
# transitively.  The renderer creates an OpenGL context through SFML and then
# uses the OpenGL API directly, so OpenGL is a real link-time dependency.
find_package(OpenGL REQUIRED)

# Prefer a system installation of SFML 3 if one is available.  Otherwise,
# FetchContent downloads the pinned version at configure time.  This keeps the
# repository small while making a fresh checkout straightforward to build.
find_package(SFML 3 COMPONENTS Graphics Window System CONFIG QUIET)

if (NOT SFML_FOUND)
    include(FetchContent)

    set(SFML_BUILD_AUDIO OFF CACHE BOOL "Disable SFML audio for this project" FORCE)
    set(SFML_BUILD_NETWORK OFF CACHE BOOL "Disable SFML network for this project" FORCE)
    set(SFML_BUILD_EXAMPLES OFF CACHE BOOL "Disable SFML examples" FORCE)
    set(SFML_BUILD_TEST_SUITE OFF CACHE BOOL "Disable SFML tests" FORCE)

    FetchContent_Declare(
        SFML
        GIT_REPOSITORY https://github.com/SFML/SFML.git
        GIT_TAG 3.0.2
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(SFML)
endif()

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------
add_executable(kerr_black_hole
    src/main.cpp
    src/particle.cpp
    src/renderer.cpp
    src/integrator.cpp
)

target_include_directories(kerr_black_hole PRIVATE src)

# SFML::Graphics brings in Window/System and SFML's OpenGL context support.
# OpenGL::GL is linked explicitly because renderer.cpp calls OpenGL directly.
target_link_libraries(kerr_black_hole PRIVATE
    SFML::Graphics
    SFML::Window
    SFML::System
    OpenGL::GL
)

# The simulation has separate AVX2 and AVX-512 implementations selected by
# preprocessor macros in utils.h.  Select the instruction set at configure time:
#
#   cmake -S . -B build -DBLACK_HOLE_SIMD=AVX2
#   cmake -S . -B build -DBLACK_HOLE_SIMD=AVX512
#
# AVX2 is the safer default.  AVX-512 should only be selected when the target
# CPU and operating system support it.  Do not use a generic "native" setting
# here: the explicit option makes the generated binary's CPU requirements clear.
set(BLACK_HOLE_SIMD "AVX2" CACHE STRING "SIMD implementation: AVX2 or AVX512")
set_property(CACHE BLACK_HOLE_SIMD PROPERTY STRINGS AVX2 AVX512)

if (BLACK_HOLE_SIMD STREQUAL "AVX2")
    target_compile_definitions(kerr_black_hole PRIVATE BLACK_HOLE_SIMD_AVX2=1)

    if (MSVC)
        # /arch:AVX2 enables AVX/AVX2 code generation on MSVC.  The existing
        # source also detects FMA separately; MSVC's AVX2 target enables the
        # relevant fused instructions on supported toolchains.
        target_compile_options(kerr_black_hole PRIVATE /arch:AVX2)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(kerr_black_hole PRIVATE -mavx2 -mfma)
    endif()
elseif (BLACK_HOLE_SIMD STREQUAL "AVX512")
    target_compile_definitions(kerr_black_hole PRIVATE BLACK_HOLE_SIMD_AVX512=1)

    if (MSVC)
        # MSVC does not expose a single /arch switch covering every AVX-512
        # feature used by this source.  Newer MSVC toolchains accept /arch:AVX512.
        target_compile_options(kerr_black_hole PRIVATE /arch:AVX512)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        # The code uses AVX-512F plus mask operations.  -mavx512f is the minimum
        # useful baseline for the implementation in utils.h.
        target_compile_options(kerr_black_hole PRIVATE -mavx512f)
    endif()
else()
    message(FATAL_ERROR "BLACK_HOLE_SIMD must be AVX2 or AVX512")
endif()

# Keep compiler warnings useful without making warnings fatal: this project
# originated as an OpenFrameworks application and some third-party/compiler
# differences are expected during the migration.
if (MSVC)
    target_compile_options(kerr_black_hole PRIVATE /W4)
else()
    target_compile_options(kerr_black_hole PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Copy the simulation input next to the executable.  The program also searches
# the source/current directory, but having the asset beside the executable is
# convenient for both Visual Studio and command-line builds.
add_custom_command(TARGET kerr_black_hole POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/data.txt"
            "$<TARGET_FILE_DIR:kerr_black_hole>/data.txt"
)
