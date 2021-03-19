#include "poisson_mnist.h"
#include ".hpp"

int main (int argc, char** argv) 
{
    // Set number of openMP threads.
    int n_threads = omp_get_max_threads();
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
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    MNISTPoison mnist_poisson(1000, 15, 1000, 20, dataset.test_images, dataset.training_images);
    mnist_poisson.GenerateAndWrite();
}
