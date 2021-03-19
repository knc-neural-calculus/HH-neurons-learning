GDB=-ggdb 
OPENMP=-fopenmp
EIGEN=-I ../eigen/
BOOST=-lboost_system -lboost_filesystem
OPTIMIZE=-O3
CFLAGS=-std=c++11 $(EIGEN) $(OPENMP) $(BOOST) $(OPTIMIZE)

hh_train : train.o
	g++ train.o -o hh_train $(CFLAGS)
	rm -rf train.o

single : train_single_case.o
	g++ train_single_case.o -o single $(CFLAGS)
	rm -rf train_single_case.o

hh_psweep : train_psweep.o
	g++ train_psweep.o -o hh_psweep $(CFLAGS)
	rm -rf train_psweep.o

gen_volt : gen_voltages.o
	g++ gen_voltages.o -o gen_volt $(CFLAGS)
	rm -rf gen_voltages.o

hh_benchmark : benchmark.o
	g++ benchmark.o -o hh_benchmark $(CFLAGS)
	rm -rf benchmark.o

poisson : poisson.o 
	g++ poisson.o -o poisson $(CFLAGS)
	rm -rf poisson.o
	
%.o: %.cpp
	g++ $(CFLAGS) -o $@ -c $<

clean : 
	rm -f *.o hh_benchmark hh_train single poisson hh_psweep gen_volt
