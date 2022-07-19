#include <iostream>
#include <algorithm>
#include <chrono>
#include "util.hpp"
#include "mergeSort.hpp"
#include <mpi/mpi.h>
#include <cmath>

using namespace std;


void sortMPI(int localSize, int id, const vector<int>& input, vector<int>& out){
    vector<int> localIn(localSize);
    cout<<"[Proc "<<id<<"]: "<<"MergeSort MPI:\n";
    MPI_Scatter(input.data(), localIn.size(), MPI_INT, localIn.data(), localIn.size(), MPI_INT, 0, MPI_COMM_WORLD);
    mergeSort(localIn.begin(), localIn.end());
    MPI_Gather(localIn.data(), localIn.size(), MPI_INT, out.data(), localIn.size(), MPI_INT, 0, MPI_COMM_WORLD);
}

void mergeMPI(int localSize, int id, int numProcs, vector<int>& out){
    vector<int> out2(localSize*2);
    vector<int> counts(numProcs), offsets(numProcs);
    for (int i=0; i < numProcs; i++){counts[i] = (i%2 == 0?localSize*2:0); offsets[i] = i*localSize;}
    MPI_Scatterv(out.data(), &counts[0], &offsets[0], MPI_INT, out2.data(), counts[id], MPI_INT, 0, MPI_COMM_WORLD);
    if(id%2==0){
        std::inplace_merge(out2.begin(), out2.begin()+localSize, out2.end());
    }
    MPI_Gatherv(out2.data(), counts[id], MPI_INT, out.data(), &counts[0], &offsets[0], MPI_INT, 0, MPI_COMM_WORLD);
    if(id==0){
        for(int i=1; i<numProcs/2; i++){
            int mid = i*localSize*2;
            int end = (i+1)*localSize*2;
            std::inplace_merge(out.begin(), out.begin()+mid, out.begin()+end);
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    int numProcs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    int id = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    auto startP = chrono::steady_clock::now();
    int n = 1000;
    if(argc>1){
        n = strtol(argv[1], NULL, 10);
    }
    vector<int> input;
    if(id == 0){
        cout<<"[Proc "<<id<<"]: "<<"Gerando array de tamanho N = "<<n<<"\n";
        input = generateInput(n);
    }
    {
        int localSize = n/numProcs;
        vector<int> out(n);
        auto start = chrono::steady_clock::now();
        sortMPI(localSize, id, input, out);
        mergeMPI(localSize, id, numProcs, out);
        auto end = chrono::steady_clock::now();
        cout<<"[Proc "<<id<<"]: "<<"Duração da ordenação: "<<(end - start)/1.0s<<"s\n";
        if(id==0) check(out);
    }
    if(id==0){
        cout<<"[Proc "<<id<<"]: "<<"MergeSort sequencial:\n";
        {
            auto start = chrono::steady_clock::now();
            mergeSort(input.begin(), input.end());
            auto end = chrono::steady_clock::now();
            cout<<"[Proc "<<id<<"]: "<<"Duração da ordenação: "<<(end - start)/1.0s<<"s\n";
            check(input);
        }
    }
    auto endP = chrono::steady_clock::now();
    cout<<"[Proc "<<id<<"]: "<<"Duração do programa: "<<(endP - startP)/1.0s<<"s\n";
    cout<<"[Proc "<<id<<"]: "<<"Ok"<<endl;
    MPI_Finalize();
    return 0;
}
