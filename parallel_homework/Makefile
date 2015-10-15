CC = icpc
OMPFLAG = -openmp
MPICC = mpicxx
NVCC = nvcc

all : toupper_serial toupper_omp toupper_mpi toupper_cuda histogram_serial histogram_omp histogram_mpi histogram_cuda

histogram_serial : histogram_serial.c
	$(CC) -I/opt/intel/tbb/2.2/include -L/opt/intel/tbb/2.2/lib/intel64 -o $@ $? -ltbb 

histogram_omp : histogram_omp.c
	$(CC) $(OMPFLAG) -I/opt/intel/tbb/2.2/include -L/opt/intel/tbb/2.2/lib/intel64 -o $@ $? -ltbb 

histogram_mpi : histogram_mpi.c
	$(MPICC) -o $@ $?

histogram_cuda : histogram_cuda.cu
	$(NVCC) -o $@ $? -arch compute_12 

toupper_serial : toupper_serial.c
	$(CC) -I/opt/intel/tbb/2.2/include -L/opt/intel/tbb/2.2/lib/intel64 -o $@ $? -ltbb 

toupper_omp : toupper_omp.c
	$(CC) $(OMPFLAG) -I/opt/intel/tbb/2.2/include -L/opt/intel/tbb/2.2/lib/intel64 -o $@ $? -ltbb 

toupper_mpi : toupper_mpi.c
	$(MPICC) -o $@ $?

toupper_cuda : toupper_cuda.cu
	$(NVCC) -o $@ $? -arch compute_12

clean : 
	rm histogram_serial
	rm histogram_omp
	rm histogram_mpi
	rm histogram_cuda
	rm toupper_serial
	rm toupper_omp
	rm toupper_mpi
	rm toupper_cuda
