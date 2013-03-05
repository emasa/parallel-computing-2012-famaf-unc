CXX=nvcc
CXXFLAGS=-O -arch=sm_20 -use_fast_math --ptxas-options "-v" --compiler-options "-Wall -O3 -march=native"
CPPFLAGS=-I/opt/cudasdk/C/common/inc

LDFLAGS=-lm

TARGETS=mm_slow.out

all: $(TARGETS)

mm_slow.out: mm_slow.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
