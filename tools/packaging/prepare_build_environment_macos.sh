#! /bin/bash

set -e
set -x

echo "CUDA is not available on MacOS"

echo "Installing libomp"
brew install llvm libomp

sudo ln -s /opt/local/include/libomp/omp.h /opt/local/include/omp.h
