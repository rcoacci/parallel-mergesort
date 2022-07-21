#include <iostream>
#include <algorithm>
#include <chrono>
#include "util.hpp"
#include "mergeSort.hpp"
#include <mpi/mpi.h>
#include <cmath>

using namespace std;

static chrono::steady_clock::duration times[4] = {};

void sortMPI(int localSize, int id, const vector<int>& input, vector<int>& out){
    vector<int> localIn(localSize);
    MPI_Scatter(input.data(), localIn.size(), MPI_INT, localIn.data(), localIn.size(), MPI_INT, 0, MPI_COMM_WORLD);
    auto start = chrono::steady_clock::now();
    mergeSort(localIn.begin(), localIn.end());
    times[0] = (chrono::steady_clock::now() - start);
    MPI_Gather(localIn.data(), localIn.size(), MPI_INT, out.data(), localIn.size(), MPI_INT, 0, MPI_COMM_WORLD);
}

void mergeMPI(int localSize, int id, int numProcs, vector<int>& out){
    vector<int> out2(localSize*2);
    vector<int> counts(numProcs), offsets(numProcs);
    for (int i=0; i < numProcs; i++){counts[i] = (i%2 == 0?localSize*2:0); offsets[i] = i*localSize;}
    MPI_Scatterv(out.data(), &counts[0], &offsets[0], MPI_INT, out2.data(), counts[id], MPI_INT, 0, MPI_COMM_WORLD);
    if(id%2==0){
        auto start = chrono::steady_clock::now();
        std::inplace_merge(out2.begin(), out2.begin()+localSize, out2.end());
        times[1] = (chrono::steady_clock::now() - start);
    }
    MPI_Gatherv(out2.data(), counts[id], MPI_INT, out.data(), &counts[0], &offsets[0], MPI_INT, 0, MPI_COMM_WORLD);
    if(id==0){
        auto start = chrono::steady_clock::now();
        for(int i=1; i<numProcs/2; i++){
            int mid = i*localSize*2;
            int end = (i+1)*localSize*2;
            std::inplace_merge(out.begin(), out.begin()+mid, out.begin()+end);
        }
        times[2] = (chrono::steady_clock::now() - start);
    }
}

int main(int argc, char *argv[])
{
    auto startP = chrono::steady_clock::now();
    int n = 1000;
    if(argc>1){
        n = strtol(argv[1], NULL, 10);
    }
    vector<int> input;

    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    int numProcs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    int id = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if(id == 0){
        cout<<"[Proc "<<id<<"]: "<<"Gerando array de tamanho N = "<<n<<"\n";
        cout<<"[Proc "<<id<<"]: ";
        input = generateInput(n);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    {
        cout<<"[Proc "<<id<<"]: "<<"MergeSort MPI:\n";
        int localSize = n/numProcs;
        vector<int> out(n);
        auto start = chrono::steady_clock::now();
        sortMPI(localSize, id, input, out);
        mergeMPI(localSize, id, numProcs, out);
        times[3] = (chrono::steady_clock::now() - start);
        if(id==0) {
            cout<<"[Proc "<<id<<"]: ";
            check(out);
        }
    }
    cout<<"[Proc "<<id<<"]: "<<"Tempo sort:      "<<times[0]/1.0s<<" s\n";
    cout<<"[Proc "<<id<<"]: "<<"Tempo merge1:    "<<times[1]/1.0s<<" s\n";
    cout<<"[Proc "<<id<<"]: "<<"Tempo merge2:    "<<times[2]/1.0s<<" s\n";
    cout<<"[Proc "<<id<<"]: "<<"Tempo total MPI: "<<times[3]/1.0s<<" s\n";
    cout<<"[Proc "<<id<<"]: "<<"Tempo processamento: "<<(times[0]+times[1]+times[2])/1.0s<<" s\n";
    cout<<"[Proc "<<id<<"]: "<<"Tempo overhead:      "<<(times[3]-times[0]-times[1]-times[2])/1.0s<<" s\n\n";
    if(id==0){
        cout<<"[Proc "<<id<<"]: "<<"MergeSort sequencial:\n";
        {
            auto start = chrono::steady_clock::now();
            mergeSort(input.begin(), input.end());
            auto end = chrono::steady_clock::now();
            cout<<"[Proc "<<id<<"]: "<<"Duração da ordenação: "<<(end - start)/1.0s<<"s\n";
            cout<<"[Proc "<<id<<"]: ";
            check(input);
        }
    }
    MPI_Finalize();
    auto endP = chrono::steady_clock::now();
    if(id==0){
        cout<<"[Proc "<<id<<"]: "<<"Duração do programa: "<<(endP - startP)/1.0s<<"s\n";
        cout<<"[Proc "<<id<<"]: "<<"Ok"<<endl;
    }
    return 0;
}
