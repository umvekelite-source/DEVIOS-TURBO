#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

namespace devios {

#pragma pack(push, 1)
struct CorpusHeader {
    char magic[8];        // "DEVIOSTB"
    uint32_t version;     // 1
    uint32_t vocab_hash;  // Hash of model vocabulary
    uint64_t n_tokens;    // Number of tokens in corpus
    uint64_t n_files;     // Number of indexed files
};
#pragma pack(pop)

struct MatchResult {
    int64_t corpus_offset; // Starting offset in tokens
    size_t match_len;      // Length of matched prefix
};

class CorpusIndex {
public:
    CorpusIndex();
    ~CorpusIndex();

    void clear();

    // Suffix array construction
    void build(const std::vector<uint32_t> & token_stream, uint32_t vocab_hash, uint64_t n_files = 1);

    // Persistence
    bool save(const std::string & file_path) const;
    bool load(const std::string & file_path, uint32_t expected_vocab_hash = 0);

    // Query: finds longest common prefix with query tokens
    MatchResult find_longest_match(const uint32_t * query, size_t qlen, size_t min_match_len = 3) const;

    // Get up to K continuation tokens starting after the match
    std::vector<uint32_t> get_draft(const MatchResult & match, size_t k) const;

    size_t size() const { return tokens.size(); }
    size_t memory_bytes() const;
    uint32_t get_vocab_hash() const { return header.vocab_hash; }

private:
    CorpusHeader header;
    std::vector<uint32_t> tokens;
    std::vector<uint32_t> sa; // Suffix array: indices into tokens

    void build_sa_doubling();
};

} // namespace devios
