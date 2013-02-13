TARGET=demo
CC=clang
CFLAGS=-O3 -Wall
LDFLAGS=-lGL -lGLU -lglut

SOURCES=$(shell echo *.c)
OBJECTS=$(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS) $(COMMON_OBJECTS)
	$(CC) $^ -o $@ $(CFLAGS) $(LDFLAGS)

clean :
	rm -f $(TARGET) $(OBJECTS) $(COMMON_OBJECTS) .depend *~

run: $(TARGET)
	./$(TARGET)

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
