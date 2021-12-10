IDIR=../include
ODIR=obj

CC=gcc
LIBS=-lm -fopenmp -lstdc++
CFLAGS=-I$(IDIR) -Wall -Wextra -pedantic 

_DEPS = csv.h data.h kmeans.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o csv.o data.o kmeans.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

../kmeans: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o