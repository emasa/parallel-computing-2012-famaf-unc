CXX=nvcc
CXXFLAGS=-O -arch=sm_20 -use_fast_math --ptxas-options "-v" --compiler-options "-Wall -O3 -march=native"
CPPFLAGS=-I/usr/local/cuda/sdk/C/common/inc

LDFLAGS=-lm -lcublas

TARGETS=mm_cublas

all: $(TARGETS)

mm_cublas: mm_cublas.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
