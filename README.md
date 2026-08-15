# Kerr black-hole simulation — SFML/OpenGL port

This is a migration of the original OpenFrameworks visualisation to **SFML 3 + OpenGL**.

The numerical integrator and message-passing architecture are deliberately kept separate from the renderer. SFML owns the window, events, timing and OpenGL context; OpenGL performs the 3D drawing.

## Build

The project uses CMake and C++20. A system SFML 3 installation is preferred; if none is found, CMake fetches SFML 3.0.2 automatically.

```bash
cmake -S . -B build -DBLACK_HOLE_SIMD=AVX2
cmake --build build --config Release
```

For an AVX-512 build:

```bash
cmake -S . -B build -DBLACK_HOLE_SIMD=AVX512
cmake --build build --config Release
```

The executable accepts the Kerr spin parameter as its first argument and an optional path to `data.txt` as its second argument:

```text
kerr_black_hole 0.3
kerr_black_hole 0.3 /path/to/data.txt
```

## OpenGL

`CMakeLists.txt` explicitly calls `find_package(OpenGL REQUIRED)` and links `OpenGL::GL`. The application requests a depth buffer, stencil buffer and multisampling through `sf::ContextSettings` before creating the SFML window.

The renderer uses OpenGL's fixed-function pipeline intentionally: this keeps the first migration close to the original OpenFrameworks implementation (`ofSpherePrimitive`, `ofLight`, `ofMaterial`, depth testing and an `ofEasyCam`-style view). It can be replaced with a modern shader/VBO renderer later without touching the integrator.

## Important SIMD note

The original source used aligned SIMD loads/stores against `std::vector` storage. `std::vector` does not generally guarantee 32/64-byte alignment for its dynamic buffer, so this port changes those operations to the corresponding unaligned intrinsics. This preserves the SIMD implementation without relying on undefined alignment assumptions.
