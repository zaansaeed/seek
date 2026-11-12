#include "text/tokenizer.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace seek {
namespace {

using Tokens = std::vector<std::string>;

TEST(Tokenize, SplitsOnPunctuationAndLowercases) {
    EXPECT_EQ(tokenize("Hello, World!"), (Tokens{"hello", "world"}));
}

TEST(Tokenize, EmptyTextGivesNoTokens) {
    EXPECT_TRUE(tokenize("").empty());
}

TEST(Tokenize, OnlySeparatorsGivesNoTokens) {
    EXPECT_TRUE(tokenize("  ,.;!?  \n\t").empty());
}

TEST(Tokenize, KeepsNumbers) {
    EXPECT_EQ(tokenize("error 503"), (Tokens{"error", "503"}));
}

TEST(Tokenize, LastTokenWithoutTrailingSeparator) {
    EXPECT_EQ(tokenize("retry upload"), (Tokens{"retry", "upload"}));
}

TEST(Tokenize, UnderscoreIsASeparatorForNow) {
    // Placeholder behaviour: step 2 decides how identifiers are handled per preset.
    EXPECT_EQ(tokenize("upload_with_retry"), (Tokens{"upload", "with", "retry"}));
}

}  // namespace
}  // namespace seek
