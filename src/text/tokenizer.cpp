#include "text/tokenizer.h"

#include <utility>  // std::move

namespace seek {

namespace {  // anonymous namespace: these helpers are private to this file

bool is_token_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

char to_lower_ascii(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

}  // namespace

std::vector<std::string> tokenize(std::string_view text) {
    std::vector<std::string> tokens;
    std::string current;

    for (char c : text) {
        if (is_token_char(c)) {
            current.push_back(to_lower_ascii(c));
        } else if (!current.empty()) {
            // std::move hands current's buffer to the vector instead of copying it;
            // current is left empty and reusable.
            tokens.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) {
        tokens.push_back(std::move(current));  // the last token, if the text didn't end with a separator
    }
    return tokens;
}

}  // namespace seek
