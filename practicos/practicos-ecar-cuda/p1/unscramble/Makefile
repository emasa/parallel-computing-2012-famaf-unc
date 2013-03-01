CXX=nvcc
CXXFLAGS=-g -G -arch=sm_20
CPPFLAGS=-I/usr/local/cuda/sdk/C/common/inc

LDFLAGS=$(shell sdl-config --libs) -lSDL_image

TARGETS=unscramble

all: $(TARGETS)

../../common/sdlstuff.o:
	make -C ../../common

unscramble: unscramble.o ../../common/sdlstuff.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

scramble: scramble.o ../../common/sdlstuff.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)


%.o: %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

.PHONY: clean all

clean:
	rm -f *.o unscramble scramble
