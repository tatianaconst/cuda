SOURCES=cuda_equation.cpp params.cpp main.cpp 
EXECUTABLE_CUDA=t3
ARCH=-arch=sm_60
HOST_COMP=mpixlC

cuda: 
	nvcc -rdc=true $(ARCH) -ccbin $(HOST_COMP) -Xcompiler -O3,-qarch=pwr8,-qstrict,-Wall cuda_param.cu $(SOURCES) -o $(EXECUTABLE_CUDA)

cudaall:
	nvcc -rdc=true $(ARCH) -x cu -ccbin $(HOST_COMP) -Xcompiler -O3,-qarch=pwr8,-qstrict,-Wall cuda_param.cu $(SOURCES) -o $(EXECUTABLE_CUDA)
