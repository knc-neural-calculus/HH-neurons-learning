#include "network.h"
#include <chrono>
using namespace std::chrono; 

int main (int argc, char** argv) 
{
    // Set number of openMP threads.
    int n_threads = 8;
    if(argc > 1)
    {
        n_threads = atoi(argv[1]);
        std::cout << "Set number of threads to " << n_threads << std::endl; 
    }
    else 
    {
        std::cout << "  WARNING: number of threads not specified. Defaulting to 8." << std::endl;
    }
    omp_set_num_threads(n_threads);
    Eigen::setNbThreads(n_threads);

    // Setup network.
    int const n0{28 * 28}, n1{100}, n2{10};
    double const dt{0.1}, sim_time_ms{300 * dt};
    NetworkBase net(n0, n1, n2, dt, sim_time_ms, 0);

    // Run simulations and do timing.
    for(int i = 0; i < 100; ++i)
    {
        auto start = high_resolution_clock::now(); 
        net.FeedForward();
        auto stop = high_resolution_clock::now(); 
        std::cout << duration_cast<milliseconds>(stop - start).count() << std::endl;
    }
}
