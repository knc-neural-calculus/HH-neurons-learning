#define DO_SAVE_WEIGHTS

#include <fstream>

#include "mnist/read_n_mnist.h"
#include "mnist/poisson_mnist.h"
#include "mnist/mnist_reader.hpp"

std::vector<uint8_t> ReadLabels (std::string fl_name)
{
    std::ifstream fl(fl_name);
    std::vector<uint8_t> labels;
    labels.reserve(60000);
    char comma;
    while(fl)
    {
        labels.push_back(0);
        fl >> labels.back() >> comma;
    }    
    fl.close();
    return labels;
}

#define __USE_POISSON__ // Toggle off to use N-MNIST

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
    Eigen::setNbThreads(n_threads);

    // Setup network.
    int const n0{28 * 28}, n1{10}, n2{10};
    double const dt{0.03}, sim_time_ms{1000 * dt};

    double const learning_rate{100};
    int const mini_batch_size{1}, n_epochs{300};
    BackpropNetwork net(n0, n1, n2, dt, sim_time_ms, learning_rate);

    net.dirname = net.gen_dirname();
    net.save_setup();
    net.open_loss_file();

    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

#ifdef __USE_POISSON__
    PoissonSampler sampler(std::move(dataset.training_labels), 1, false);
    net.SGD(mini_batch_size, n_epochs, &sampler);
    net.EvaluateNetwork(&sampler, 1000);
#else 
    NMNISTReader reader("../../N-MNIST-TRAIN/", std::move(dataset.training_labels));
    net.SGD(mini_batch_size, n_epochs, &reader);
#endif // __USE_POISSON__
}
