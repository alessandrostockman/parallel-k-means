IDIR=../include
ODIR=obj

CC=gcc
LIBS=-lm -lstdc++
CFLAGS=-I$(IDIR) -Wall -Wextra -pedantic -fopenmp

_DEPS = csv.h data.h kmeans.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o csv.o data.o kmeans.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

../kmeans.exe: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o
	rm ../*.exe