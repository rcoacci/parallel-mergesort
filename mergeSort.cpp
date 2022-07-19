#include <algorithm>
#include "mergeSort.hpp"

using namespace std;

void mergeSort(vector<int>::iterator in1, vector<int>::iterator in2){
    if (in2 - in1 > 1){
        int mid = (in2-in1)/2;
        mergeSort(in1, in1+mid);
        mergeSort(in1+mid, in2);
        std::inplace_merge(in1, in1+mid, in2);
    }
}
