make:
	mpic++ -O2 -std=c++17 -o simulate main.cpp
run:
	mpirun -np 4 ./simulate steps.txt
