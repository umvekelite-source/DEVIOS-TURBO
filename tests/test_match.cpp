#include "devios_corpus.h"
#include <iostream>
#include <cassert>

int main() {
    devios::CorpusIndex idx;
    // Corpus: "void quicksort(vector<int> & arr, int low, int high) { if (low < high) { int pivot = arr[high]; } }"
    // represented as tokens 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    std::vector<uint32_t> tokens = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130 };
    idx.build(tokens, 12345);

    // Query ending with: 30, 40, 50
    std::vector<uint32_t> q1 = { 999, 888, 30, 40, 50 };
    auto m1 = idx.find_longest_match(q1.data(), q1.size(), 3);
    std::cout << "Match m1: offset=" << m1.corpus_offset << " len=" << m1.match_len << std::endl;
    auto draft1 = idx.get_draft(m1, 4);
    std::cout << "Draft: ";
    for (auto d : draft1) std::cout << d << " ";
    std::cout << "\n";

    // Expected: match at offset 2, len 3 (tokens 30, 40, 50). Continuation: 60, 70, 80, 90.
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
