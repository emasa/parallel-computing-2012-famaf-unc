CC=gcc
CFLAGS=-O3 -march=native -std=c99 -Wall -fopenmp

# CC=icc
# CFLAGS=-fast -std=c99 -openmp

LDFLAGS=-lm

TARGETS=parallel_membw schedule overhead falsesharing

SOURCES=$(shell echo *.c)
OBJECTS=$(SOURCES:.c=.o)

all: $(TARGETS)


parallel_membw: parallel_membw.o
	$(CC) $(CFLAGS) -o $@ $^

schedule: schedule.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

overhead: overhead.o MCMLrng.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

falsesharing: falsesharing.o MCMLrng.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJECTS) $(TARGETS)

.PHONY: clean all
