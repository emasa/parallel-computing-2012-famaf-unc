CC=gcc
#CC=clang
#CC=icc

# probando flags de vectorizacion automatica, loop unrolling automatico, ffast-math
CFLAGS=-O3 -Wall -march=native -std=c99 -ftree-vectorizer-verbose=4 -fprefetch-loop-arrays -funroll-loops -fvect-cost-model -ffast-math -msse4a
#CFLAGS=-O3 -Wall -march=native -std=c99 -ftree-vectorizer-verbose=4 -fprefetch-loop-arrays -funroll-loops -fvect-cost-model -msse4a
#CFLAGS=-O3 -Wall -march=native -std=c99 -ffast-math -msse4a
LDFLAGS=

TARGETS=headless.out demo.out
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o
TARGETS_OBJECTS=headless.o demo.o

all: $(TARGETS)

demo.out: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless.out: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS) $(COMMON_OBJECTS) $(TARGETS_OBJECTS) .depend *~

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
