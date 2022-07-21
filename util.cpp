#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include "util.hpp"

using namespace std;

void print(const int& n){
    std::cout << " " << n;
};

vector<int> generateInput(size_t n){

    auto start = chrono::steady_clock::now();
    random_device rd;  //Will be used to obtain a seed for the random number engine
    default_random_engine gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    uniform_int_distribution<> distrib(1, 100*n);
    vector<int> input(n);
    generate(input.begin(), input.end(), [&](){return distrib(gen);});
    auto end = chrono::steady_clock::now();
    cout<<"Duração da geração do array: "<<(end - start)/1.0s<<"s\n";
    if(n<100){
        cerr<<"Array original:\n";
        for_each(input.begin(), input.end(), print);
        cerr<<endl;
    }
    return input;
}

void check(const vector<int>& out){
    auto start = chrono::steady_clock::now();
    auto ok = is_sorted(out.begin(), out.end());
    auto end = chrono::steady_clock::now();
    cout<<"Duração da verificação: "<<(end - start)/1.0s<<"s\n";
    if(!ok){
        cout<<"Array resposta nao esta ordenado corretamente!\n";
        std::exit(-1);
    }
}
