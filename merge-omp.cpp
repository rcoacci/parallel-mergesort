#include <iostream>
#include <algorithm>
#include <chrono>
#include "util.hpp"
#include "mergeSort.hpp"

using namespace std;

template<typename InputIter, typename OutputIter>
void mergeSortOMP(InputIter in1, InputIter in2, OutputIter out1, size_t minSize){
    std::size_t size = in2 - in1;
    if(size>1){
        InputIter mid = in1 + (size/2);
        OutputIter out2 = out1 + (size/2);
        OutputIter out3 = out1 + size;
        #pragma omp task untied depend(out: out1) if(size>minSize)
        mergeSortOMP(in1, mid, out1, minSize);
        #pragma omp task untied depend(out: out2) if(size>minSize)
        mergeSortOMP(mid, in2, out2, minSize);
        #pragma omp task untied depend(in:out1, out2) if(size>minSize)
        std::inplace_merge(out1,out2,out3);
        #pragma omp taskwait
    } else {
        *out1 = *in1;
    }
}

template<typename Container>
Container parallelMergeSortOMP(Container& in, size_t minSize = 128){
    Container out(in.size());
    #pragma omp parallel master
    mergeSortOMP(in.begin(), in.end(), out.begin(), minSize);
    return out;
}

int main(int argc, char *argv[])
{
    auto startP = chrono::steady_clock::now();
    int n = 1000;
    if(argc>1){
        n = strtol(argv[1], NULL, 10);
    }
    int minSize = 100;
    if(argc>2){
        minSize = strtol(argv[2], NULL, 10);
    }
    cout<<"Gerando array de tamanho N = "<<n<<"\n";
    auto input = generateInput(n);
    cout<<"MergeSort OpenMP:\n";
    {
        auto start = chrono::steady_clock::now();
        auto out = parallelMergeSortOMP(input, minSize);
        auto end = chrono::steady_clock::now();
        cout<<"Duração da ordenação: "<<(end - start)/1.0s<<"s\n";
        check(out);
    }
    cout<<"MergeSort sequencial:\n";
    {
        vector<int> out(input);
        auto start = chrono::steady_clock::now();
        mergeSort(out.begin(), out.end());
        auto end = chrono::steady_clock::now();
        cout<<"Duração da ordenação: "<<(end - start)/1.0s<<"s\n";
        check(out);
    }
    auto endP = chrono::steady_clock::now();
    cout<<"Duração do programa: "<<(endP - startP)/1.0s<<"s\n";
    cout<<"Ok"<<endl;
    return 0;
}
