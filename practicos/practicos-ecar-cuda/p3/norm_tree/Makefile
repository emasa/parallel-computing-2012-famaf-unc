CXX=nvcc
CXXFLAGS=-O -arch=sm_20
CPPFLAGS=-I/opt/cudasdk/C/common/inc

LDFLAGS=$(shell sdl-config --libs) -lSDL_image

TARGETS=norm_tree.out

all: $(TARGETS)

../../common/sdlstuff.o:
	make -C ../../common

norm_tree.out: norm_tree.o ../../common/sdlstuff.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
