#define NEURON_ARE_VARS
#define DO_SAVE_WEIGHTS
#define DONT_SET_PHYSIOLOGY

#include <unordered_map>
#include "configer.h"
#include "mnist/read_n_mnist.h"
#include "mnist/poisson_mnist.h"
#include "mnist/mnist_reader.hpp"

int main (int argc, char** argv) 
{
    // * threads

    // Set number of openMP threads.
    // int n_threads = omp_get_max_threads();
	int n_threads = 1;
    omp_set_num_threads(n_threads);
    Eigen::setNbThreads(n_threads);
    

    // * read config
    // get name of config file and read it
    std::string dirname;
	int epoch_load = 0;

	if (argc < 3)
	{
		std::cout << "dude gimme the dirname nd epoch number" << std::endl;
		std::cout << "format:" << std::endl;
		std::cout << "\t./gen_voltages <dirname> <epoch_num>" << std::endl;
		std::cout << "dont forget to NOT put a slash in the dirname" << std::endl;
		return 1;
	}

	dirname = argv[1];
	epoch_load = std::stoi(argv[2]);

	printf("loaded args:\n");
	printf("dirname =\t%s\n", dirname.c_str());
	printf("epoch =\t%d\n", epoch_load);



    // read a config file
    std::unordered_map<std::string, std::string> config = ConfigReader("../../psweep_data/" + dirname + "/config.txt");
    
    // * use config
    // update modifiable "consts"
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

    config["DIRNAME"] = "../../psweep_data/" + dirname;
    net.dirname = config["DIRNAME"];
    net.SetCostScalar(std::stof(config["OUTPUT_SCALAR"]));
    
    // print config to stdout
    std::cout << "input config:" << std::endl;
    for (auto& el : config)
    {
        std::cout << "\t" << el.first << "\t:\t" << el.second << "\n";
    }
    std::cout << std::endl;


    // * get data and run

    printf("loading epoch weights and biases:\t%d\n", epoch_load);
    net.ReadWeightsAndBiases(config["DIRNAME"], epoch_load);
    printf("loading epoch:\t%d\n", epoch_load);
    {
        auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
        PoissonSampler sampler(10, std::move(dataset.test_labels), 1, false);
	net.ToggleVoltagePrinting(true);
        net.LoadInputFromSampler(&sampler);
	net.FeedForward(false);
	net.ToggleVoltagePrinting(false);
    }
    {
        auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
        PoissonSampler test_sampler(10000, std::move(dataset.test_labels), 1, false);
        net.EvaluateNetwork(&test_sampler, epoch_load, 500);
    }
    std::cout << "Averaged output: " << net.GetOutputAverage() << std::endl;
}
