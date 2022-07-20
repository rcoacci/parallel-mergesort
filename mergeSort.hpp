#include <vector>
#include <algorithm>

/**
 * Inplace Merge Sort.
 */
template<typename Iter>
void mergeSort(Iter begin, Iter end){
    std::size_t size = end-begin;
    if (size > 1){
        Iter mid = begin+(size)/2;
        mergeSort(begin, mid);
        mergeSort(mid, end);
        std::inplace_merge(begin, mid, end);
    }
}
