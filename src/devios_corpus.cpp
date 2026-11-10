#include "devios_corpus.h"

#include <cstring>
#include <cstdio>
#include <iostream>
#include <numeric>

namespace devios {

static const char MAGIC_BYTES[8] = {'D', 'E', 'V', 'I', 'O', 'S', 'T', 'B'};

CorpusIndex::CorpusIndex() {
    std::memset(&header, 0, sizeof(header));
    std::memcpy(header.magic, MAGIC_BYTES, 8);
    header.version = 1;
}

CorpusIndex::~CorpusIndex() {
    clear();
}

void CorpusIndex::clear() {
    tokens.clear();
    tokens.shrink_to_fit();
    sa.clear();
    sa.shrink_to_fit();
    std::memset(&header, 0, sizeof(header));
    std::memcpy(header.magic, MAGIC_BYTES, 8);
    header.version = 1;
}

size_t CorpusIndex::memory_bytes() const {
    return sizeof(CorpusIndex) +
           tokens.capacity() * sizeof(uint32_t) +
           sa.capacity() * sizeof(uint32_t);
}

void CorpusIndex::build_sa_doubling() {
    const size_t n = tokens.size();
    if (n == 0) {
        sa.clear();
        return;
    }
    sa.resize(n);
    if (n == 1) {
        sa[0] = 0;
        return;
    }

    std::vector<uint32_t> rank(n);
    std::vector<uint32_t> temp_rank(n);

    for (size_t i = 0; i < n; ++i) {
        sa[i] = static_cast<uint32_t>(i);
    }

    std::sort(sa.begin(), sa.end(), [&](uint32_t a, uint32_t b) {
        return tokens[a] < tokens[b];
    });

    rank[sa[0]] = 1;
    for (size_t i = 1; i < n; ++i) {
        rank[sa[i]] = rank[sa[i - 1]] + (tokens[sa[i]] != tokens[sa[i - 1]] ? 1 : 0);
    }

    for (size_t k = 1; k < n; k <<= 1) {
        auto cmp = [&](uint32_t a, uint32_t b) {
            if (rank[a] != rank[b]) {
                return rank[a] < rank[b];
            }
            uint32_t ra = (a + k < n) ? rank[a + k] : 0;
            uint32_t rb = (b + k < n) ? rank[b + k] : 0;
            return ra < rb;
        };

        std::sort(sa.begin(), sa.end(), cmp);

        temp_rank[sa[0]] = 1;
        for (size_t i = 1; i < n; ++i) {
            temp_rank[sa[i]] = temp_rank[sa[i - 1]] + (cmp(sa[i - 1], sa[i]) ? 1 : 0);
        }
        rank = temp_rank;

        if (rank[sa[n - 1]] == n) {
            break;
        }
    }
}

void CorpusIndex::build(const std::vector<uint32_t> & token_stream, uint32_t vocab_hash, uint64_t n_files) {
    tokens = token_stream;
    header.vocab_hash = vocab_hash;
    header.n_tokens = tokens.size();
    header.n_files = n_files;
    header.version = 1;
    std::memcpy(header.magic, MAGIC_BYTES, 8);

    build_sa_doubling();
}

bool CorpusIndex::save(const std::string & file_path) const {
    std::ofstream out(file_path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    if (header.n_tokens > 0) {
        out.write(reinterpret_cast<const char *>(tokens.data()), tokens.size() * sizeof(uint32_t));
        out.write(reinterpret_cast<const char *>(sa.data()), sa.size() * sizeof(uint32_t));
    }
    return out.good();
}

bool CorpusIndex::load(const std::string & file_path, uint32_t expected_vocab_hash) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    CorpusHeader hdr;
    in.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
    if (!in.good() || std::memcmp(hdr.magic, MAGIC_BYTES, 8) != 0) {
        return false;
    }

    if (expected_vocab_hash != 0 && hdr.vocab_hash != expected_vocab_hash) {
        return false;
    }

    header = hdr;
    tokens.resize(header.n_tokens);
    sa.resize(header.n_tokens);

    if (header.n_tokens > 0) {
        in.read(reinterpret_cast<char *>(tokens.data()), header.n_tokens * sizeof(uint32_t));
        in.read(reinterpret_cast<char *>(sa.data()), header.n_tokens * sizeof(uint32_t));
    }

    return in.good();
}

MatchResult CorpusIndex::find_longest_match(const uint32_t * query, size_t qlen, size_t min_match_len) const {
    MatchResult best = {-1, 0};
    const size_t n = tokens.size();
    if (n == 0 || qlen < min_match_len) {
        return best;
    }

    // The context tail ending at qlen - 1 must match the corpus substring.
    // Try lengths L from max_len down to min_match_len.
    // The first match of length L found is guaranteed to be the longest match.
    size_t max_l = std::min<size_t>(qlen, 32);

    for (size_t L = max_l; L >= min_match_len; --L) {
        const uint32_t * sub_q = query + (qlen - L);

        int64_t low = 0;
        int64_t high = static_cast<int64_t>(n) - 1;

        while (low <= high) {
            int64_t mid = low + (high - low) / 2;
            uint32_t pos = sa[mid];

            // Compare sub_q[0..L-1] with tokens[pos..pos+L-1]
            int cmp = 0;
            for (size_t i = 0; i < L; ++i) {
                if (pos + i >= n) {
                    cmp = 1; // Suffix too short, sub_q is greater
                    break;
                }
                if (sub_q[i] != tokens[pos + i]) {
                    cmp = (sub_q[i] > tokens[pos + i]) ? 1 : -1;
                    break;
                }
            }

            if (cmp == 0) {
                // Exact match of length L that matches the context tail!
                best.corpus_offset = pos;
                best.match_len = L;
                return best;
            } else if (cmp > 0) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }

    return best;
}

std::vector<uint32_t> CorpusIndex::get_draft(const MatchResult & match, size_t k) const {
    std::vector<uint32_t> draft;
    if (match.corpus_offset < 0 || match.match_len == 0 || k == 0) {
        return draft;
    }

    size_t draft_start = static_cast<size_t>(match.corpus_offset) + match.match_len;
    size_t count = std::min(k, tokens.size() > draft_start ? tokens.size() - draft_start : 0);

    draft.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        draft.push_back(tokens[draft_start + i]);
    }

    return draft;
}

} // namespace devios
