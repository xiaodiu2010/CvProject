CC=g++
CFLAGS=-fPIC

all: libcgco.so

libcgco.so: LinkedBlockList.o graph.o maxflow.o GCoptimization.o cgco.o
	$(CC) -shared $(CFLAGS) LinkedBlockList.o graph.o maxflow.o GCoptimization.o cgco.o -o libcgco.so

gco.so: LinkedBlockList.o graph.o maxflow.o GCoptimization.o
	$(CC) -shared $(CFLAGS) LinkedBlockList.o graph.o maxflow.o GCoptimization.o -o gco.so

LinkedBlockList.o: LinkedBlockList.cpp LinkedBlockList.h
	$(CC) $(CFLAGS) -c LinkedBlockList.cpp -o LinkedBlockList.o

graph.o: graph.cpp graph.h block.h
	$(CC) $(CFLAGS) -c graph.cpp -o graph.o

maxflow.o: block.h graph.h maxflow.cpp
	$(CC) $(CFLAGS) -c maxflow.cpp -o maxflow.o

GCoptimization.o: GCoptimization.cpp GCoptimization.h LinkedBlockList.h energy.h graph.h graph.o maxflow.o
	$(CC) $(CFLAGS) -c GCoptimization.cpp -o GCoptimization.o

cgco.o: cgco.cpp GCoptimization.h
	$(CC) $(CFLAGS) -c cgco.cpp -o cgco.o

test_wrapper: test_wrapper.cpp
	$(CC) -L. test_wrapper.cpp -o test_wrapper -lcgco

clean:
	rm -f *.o

rm:
	rm -f *.o *.so

