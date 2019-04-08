CXX ?= g++
CFLAGS = -Wall -Wconversion -g -fPIC
SHVER = 2
OS = $(shell uname)

all: svmmain


svm-predict.o: svm-predict.c svm.o svm-scale.o
	$(CXX) $(CFLAGS) -c svm-predict.c
svm-train.o: svm-train.c svm.o
	$(CXX) $(CFLAGS) -c svm-train.c
svm-scale.o: svm-scale.c
	$(CXX) $(CFLAGS) -c svm-scale.c
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
svmmain.o: svmmain.c
	$(CXX) $(CFLAGS) -c svmmain.c
svmmain: svmmain.o svm.o svm-predict.o svm-train.o svm-scale.o
	$(CXX) $(CFLAGS) svmmain.o svm.o svm-predict.o svm-train.o svm-scale.o -lm -o svmmain
clean:
	rm -f *o svmmain
