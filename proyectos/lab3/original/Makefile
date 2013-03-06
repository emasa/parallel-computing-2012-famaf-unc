 CC=gcc
 CFLAGS=-std=c99 -O3 -march=native -ffast-math -ftree-vectorizer-verbose=1

# CC=clang
# CFLAGS=-std=c99 -O3 -march=native

# CC=icc
# CFLAGS=-std=c99 -fast

# CC=/opt/ekopath/1.0.0/bin/pathcc
# CFLAGS=-std=c99 -Ofast -march=nehalem

LDFLAGS=

TARGETS=headless.out demo.out
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o

all: $(TARGETS)

demo.out: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless.out: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o .depend *~

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
