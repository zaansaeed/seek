// M0 spike, stage 1: confirm llama.cpp builds and links with Metal enabled.
// Stage 2 (step 5) grows this into the embedding sanity check.

#include <cstdio>

#include "llama.h"

int main() {
    llama_backend_init();

    // One line listing the accelerators compiled in; we expect Metal to appear.
    std::printf("%s\n", llama_print_system_info());

    llama_backend_free();
    return 0;
}
