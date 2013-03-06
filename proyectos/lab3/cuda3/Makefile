
CXX=nvcc
CXXFLAGS=-O -arch=sm_20 -use_fast_math --ptxas-options "-v" --compiler-options "-Wall -O3 -march=native"
CPPFLAGS=-I/opt/cudasdk/C/common/inc

LDFLAGS=-lm

TARGETS=headless.out demo.out
SOURCES=$(shell echo *.c *.cu)
COMMON_OBJECTS=timing.o solver.o

all: $(TARGETS)

demo.out: demo.o $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless.out: headless.o $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -f *.o $(TARGETS) *~

.PHONY: clean all
