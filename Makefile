CCPOL=mpixlC
POLUS_FLAGS=-DPOLUS -qarch=pwr8
CFLAGS=-O0 -qstrict
INC=
INC_PARAMS=$(foreach d, $(INC), -I$d)
LDFLAGS=-lm
SOURCES=cuda_equation.cpp params.cpp main.cpp 
EXECUTABLE_CUDA=t3
EXECUTABLE_FLOAT_SEQ=t2_seq

cuda: 
	nvcc -rdc=true -arch=sm_60 -ccbin mpixlC -Xcompiler -O0,-qarch=pwr8,-qstrict,-Wall,-DCUDA cuda_param.cu $(SOURCES) -o $(EXECUTABLE_CUDA)

cl:
	rm -f *.o *.out *.err core.*
