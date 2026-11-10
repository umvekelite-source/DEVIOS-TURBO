#include "devios_corpus.h"
#include <iostream>
#include <cassert>

int main() {
    devios::CorpusIndex idx;
    std::vector<uint32_t> tokens = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130 };
    idx.build(tokens, 12345);

    std::vector<uint32_t> q1 = { 999, 888, 30, 40, 50 };
    auto m1 = idx.find_longest_match(q1.data(), q1.size(), 3);
    std::cout << "Match m1: offset=" << m1.corpus_offset << " len=" << m1.match_len << std::endl;
    auto draft1 = idx.get_draft(m1, 4);
    std::cout << "Draft: ";
    for (auto d : draft1) std::cout << d << " ";
    std::cout << "\n";

    assert(m1.corpus_offset == 2);
    assert(m1.match_len == 3);
    assert(draft1.size() == 4);
    assert(draft1[0] == 60);
    assert(draft1[1] == 70);
    assert(draft1[2] == 80);
    assert(draft1[3] == 90);
    std::cout << "Test passed!\n";
    return 0;
}
