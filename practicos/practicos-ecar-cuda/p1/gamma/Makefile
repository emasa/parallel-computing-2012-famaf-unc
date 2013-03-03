CXX=nvcc
CXXFLAGS=-g -G -arch=sm_20
CPPFLAGS=-I/opt/cudasdk/C/common/inc/

LDFLAGS=$(shell sdl-config --libs) -lSDL_image

TARGETS=gamma.out

all: $(TARGETS)

../../common/sdlstuff.o:
	make -C ../../common

gamma.out: gamma.o ../../common/sdlstuff.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
