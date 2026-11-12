cmake_minimum_required(VERSION 3.20)
project(seek LANGUAGES C CXX)

# Write build/compile_commands.json so editor tooling (clangd) can find headers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
# llama.cpp, pinned as a git submodule. Built as part of this project; gives us the `llama` target.
add_subdirectory(third_party/llama.cpp)

# M0 spike: load the embedder and sanity-check its vectors. Throwaway; the real embedder lands in src/model/ in M2.
add_executable(spike_embed spike/embed_spike.cpp)
target_link_libraries(spike_embed PRIVATE llama)
target_compile_features(spike_embed PRIVATE cxx_std_20)

# M0 spike: run the teacher reranker and check its scores against the model card. Real reranker lands in src/model/ in M4.
add_executable(spike_rerank spike/rerank_spike.cpp)
target_link_libraries(spike_rerank PRIVATE llama)
target_compile_features(spike_rerank PRIVATE cxx_std_20)

# M0 spike: run a small BERT cross-encoder (the student's shape) and match PyTorch. Real student lands in src/model/ in M5.
add_executable(spike_student spike/student_spike.cpp)
target_link_libraries(spike_student PRIVATE llama)
target_compile_features(spike_student PRIVATE cxx_std_20)
