cmake_minimum_required(VERSION 3.25)  # 3.25+ for add_subdirectory(... SYSTEM)
project(seek LANGUAGES C CXX)

# Write build/compile_commands.json so editor tooling (clangd) can find headers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Third-party code (pinned git submodules) ----
# SYSTEM: their headers are included with -isystem, so our strict warnings don't fire on their code.

# llama.cpp: gives us the `llama` target.
add_subdirectory(third_party/llama.cpp SYSTEM)

# GoogleTest: gives us the `gtest` and `gtest_main` targets. We don't need GoogleMock or install rules.
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/googletest SYSTEM)

# ---- Shared settings for all of seek's own targets ----
# An INTERFACE library holds no code, only settings; anything that links it inherits them.
add_library(seek_options INTERFACE)
target_compile_features(seek_options INTERFACE cxx_std_20)
target_compile_options(seek_options INTERFACE
    $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra -Wpedantic>)

# ---- seek libraries (one per folder in src/) ----

# text: turning raw text into index terms (tokenizer, later the full analyzer).
add_library(seek_text
    src/text/tokenizer.cpp)
target_include_directories(seek_text PUBLIC src)  # so code writes #include "text/tokenizer.h"
target_link_libraries(seek_text PUBLIC seek_options)

# ---- Tests ----
enable_testing()
include(GoogleTest)

add_executable(seek_tests
    tests/text/tokenizer_test.cpp)
target_link_libraries(seek_tests PRIVATE seek_text gtest_main)
gtest_discover_tests(seek_tests)  # registers every TEST() with CTest, so `ctest` runs them
