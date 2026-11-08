CC     = gcc
CFLAGS = -O2 -Wall -Wextra -msse2

.PHONY: full clean

full: main
	./main

main: main.c
	$(CC) $(CFLAGS) main.c -o main -lm

clean:
	rm -f main
