#define NEURON_ARE_VARS
#define DO_SAVE_WEIGHTS
//#define DONT_SET_PHYSIOLOGY

#include <unordered_map>

#include "configer.h"
#include "mnist/read_n_mnist.h"
#include "mnist/poisson_mnist.h"
#include "mnist/mnist_reader.hpp"

int main (int argc, char** argv) 
{
    // * threads

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
    

    // * read config
    // get name of config file and read it
    std::string config_file;
    if(argc == 2)
    {
        std::cout << "\n\nFATAL ERROR: config file not specified in command line args. terminating" << std::endl;
        return 1;
    }
    else if(argc == 3)
    {
        std::cout << "\n\nFATAL ERROR: run index not specified in command line args. terminating" << std::endl;
        return 1;
    }
    else 
    {
        config_file = "psweep/config/" + std::string(argv[3]) + "_ID" + std::string(argv[2]) + ".txt";
    }

    // read a config file
    std::unordered_map<std::string, std::string> config = ConfigReader(config_file);
    
    // * use config
    // update modifiable "consts"
    bool do_transient = true;
    Neuron::a_hidden  = std::stof(config["COUPLING_HIDDEN"]);
    Neuron::a_out     = std::stof(config["COUPLING_OUT"]);
    Neuron::LF_hidden = std::stof(config["LF_HIDDEN"]);
    Neuron::LF_out    = std::stof(config["LF_OUT"]);
    Neuron::use_bias  = std::stof(config["USE_BIAS"]);
#ifndef DONT_SET_PHYSIOLOGY
    Neuron::use_beta_phase_2 = std::stof(config["BETA_PHASE_2"]);
    do_transient = Neuron::use_beta_phase_2 > 0.5;  // No transient setup for phase 2. This messes things up bc the neurons already fire and can only fire once.
    Neuron::gna = std::stof(config["GNA"]); 
    Neuron::gk = std::stof(config["GK"]);
    Neuron::gl = std::stof(config["GL"]);
    Neuron::Ena = std::stof(config["ENA"]);
    Neuron::Ek = std::stof(config["EK"]);
    Neuron::El = std::stof(config["EL"]);
#endif

    // Setup network.
    int const mini_batch_size{std::stoi(config["BATCH_SIZE"]) }, n_epochs{ std::stoi(config["MAX_EPOCHS"]) };
    BackpropNetwork net(
        std::stoi(config["N_LAYER_0"]), // n_in
        std::stoi(config["N_LAYER_1"]), // n_hidden
        std::stoi(config["N_LAYER_2"]), // n_out
        std::stof(config["DELTA_T"]), // dt
        std::stof(config["SIM_STEPS"]) * std::stof(config["DELTA_T"]), // sim_time_ms
        std::stof(config["LEARNING_RATE"]) // learning_rate
    );
    int n_sniffs = std::stoi(config["NUM_SNIFFS"]);

    // TODO : noise param

    config["DIRNAME"] = "../../psweep_data/" + config["DIRNAME"];
    net.dirname = config["DIRNAME"];
    net.SetCostScalar(std::stof(config["OUTPUT_SCALAR"]));

    // * save config
    save_setup(config);
    // print config to stdout
    std::cout << "input config:" << std::endl;
    for (auto& el : config)
    {
        std::cout << "\t" << el.first << "\t:\t" << el.second << "\n";
    }
    std::cout << std::endl;


    // * get data and run
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    net.open_loss_file();

    PoissonSampler sampler(n_epochs * mini_batch_size, std::move(dataset.training_labels), 1, true);
    PoissonSampler test_sampler(10000, std::move(dataset.test_labels), 1, false);
    for(int s = 0; s < n_sniffs; ++s)
    {
        sampler.ShuffleOrder();
        net.SGD(mini_batch_size, n_epochs, &sampler);
        sampler.Reset();
        test_sampler.Reset();
    }
    net.EvaluateNetwork(&test_sampler, 0, 100);
    // Create an empty file called DONE.txt to let them know we are done. 
    std::ofstream done_file(net.dirname + "/DONE.txt");
    done_file.close();
}
