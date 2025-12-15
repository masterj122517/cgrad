CC = gcc
CFLAGS = -Wall -Werror
TARGET = bin/cgrad 
SRCS = engine.c main.c
OBJS = $(SRCS:%.c=bin/%.o)

all: $(TARGET)

$(TARGET): $(OBJS) | bin
	$(CC) $(CFLAGS) -o $@ $^

bin/%.o: %.c | bin
	$(CC) $(CFLAGS) -c $< -o $@

bin:
	mkdir -p bin

clean:
	rm -rf bin

run: $(TARGET)
	./$(TARGET)
