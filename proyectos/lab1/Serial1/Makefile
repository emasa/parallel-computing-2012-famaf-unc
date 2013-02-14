CC=gcc
#CC=clang
#CC=icc

CFLAGS=-O3 -march=native -msse4 -Wall -std=c99 -g -pg
LDFLAGS=

TARGETS=headless demo
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS) $(COMMON_OBJECTS) .depend *~

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
