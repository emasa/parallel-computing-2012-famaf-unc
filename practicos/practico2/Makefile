CC=g++
# CC=clang
CFLAGS=-O3 -march=native -Wall

# CC=icc
# CFLAGS=-fast -std=c99

TARGET=transpose_sse

SOURCES=$(shell echo *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean all
