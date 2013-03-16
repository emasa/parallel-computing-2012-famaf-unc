CC=gcc
#CC=clang
#CC=icc

# probando flags de vectorizacion automatica, loop unrolling automatico, ffast-math 
CFLAGS=-O3 -Wall -march=native -std=c99 -ftree-vectorizer-verbose=0 -fprefetch-loop-arrays -funroll-loops -fvect-cost-model -ffast-math -msse4.2 # -g -pg
#CFLAGS=-O3 -Wall -funroll-loops -std=c99 -ffast-math -lm  #clang
#CFLAGS=-fast -Wall -march=native -std=c99 -vec-report2 -unroll-aggressive -opt-prefetch #icc

LDFLAGS=-lm

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
	$(CC) $(CFLAGS) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
