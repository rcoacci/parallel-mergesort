project('merge', ['cpp'],
  version : '0.1',
  default_options : ['warning_level=3',
                     'cpp_std=c++17',
                     'cpp_args=-Wno-unused-parameter'])

openmp = dependency('openmp')
mpi = dependency('mpi', language:'cpp')
executable('merge-omp',['merge-omp.cpp', 'util.cpp', 'mergeSort.cpp'], dependencies: openmp)
executable('merge-mpi',['merge-mpi.cpp', 'util.cpp', 'mergeSort.cpp'], dependencies: mpi)
