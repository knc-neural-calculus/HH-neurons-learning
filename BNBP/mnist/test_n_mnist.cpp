#include "read_n_mnist.h"
#include "mnist_reader.hpp"

int main () 
{
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    NMNISTReader reader("../../N-MNIST-TRAIN/", std::move(dataset.training_labels));


    int const n0{28 * 28}, n2{10};
    double const dt{0.03}, sim_time_ms{1000 * dt};
    int const sim_incs = sim_time_ms / dt;
    Mat inp(n0, sim_incs);
    Vec desired_out(n2);

    int idx = 1;
    while(true)
    {
        try
        {
            if((idx - 1) % 1== 0)
                std::cout << idx << std::endl;
            reader.LoadNextSpiketrain(inp, desired_out, dt);
            ++idx;
        }
        catch(...)
        {
            break;
        }
    }
}
