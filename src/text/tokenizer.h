#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace seek {

// Splits text into lowercase word tokens.
//
// A token is a maximal run of ASCII letters (a-z, A-Z) and digits (0-9). Every other
// character (spaces, punctuation, underscores, apostrophes, non-ASCII bytes) is a
// separator and never appears in the output. Letters are lowercased.
//
// Input:
//   text  The text to split. Read-only; a string_view, so callers can pass a
//         std::string, a string literal, or a slice of a larger buffer without copying.
//
// Returns:
//   The tokens in the order they appear in `text`. Empty if `text` has no letters or
//   digits. Repeated words are kept (counting them is the index's job).
//
// Example:
//   tokenize("Hello's World, error 503!")  ->  {"hello", "s", "world", "error", "503"}
//
// This is the simple first version. Step 2 grows it into the full analyzer
// (possessives, stopwords, Porter stemming, and identifier splitting for the `seek` preset).
std::vector<std::string> tokenize(std::string_view text);

}  // namespace seek
