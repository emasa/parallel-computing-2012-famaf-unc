CXX=nvcc
CXXFLAGS=-g -G -arch=sm_20 --ptxas-options=-v
CPPFLAGS=-I/opt/cudasdk/C/common/inc/

LDFLAGS=$(shell sdl-config --libs) -lSDL_image

TARGETS=grid2d.out

all: $(TARGETS)

../../common/sdlstuff.o:
	make -C ../../common

grid2d.out: grid2d.o ../../common/sdlstuff.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
