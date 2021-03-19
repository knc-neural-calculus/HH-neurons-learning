#pragma once

#include <fstream>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <iostream>


std::unordered_map<std::string, std::string> ConfigReader(std::string config_file)
{
    std::cout << "reading config file:\t" << config_file << std::endl;
    std::unordered_map<std::string, std::string> data;
    std::ifstream fin(config_file);

    if (!fin)
    {
        throw std::runtime_error("failed to open '" + config_file + "' for reading");
    }

    std::string key,junk,val;

    while (!fin.eof())
    {
        fin >> key >> junk >> val;
        // std::cout << "\t" << key << "\t" << junk << "\t" << val << std::endl;
        data[key] = val;
    }

    fin.close();
    return data;
}

// saves metadata of this network, including some parameters from `hh.h`
void save_metadata(std::unordered_map<std::string, std::string> config)
{
    std::ofstream fout;

    fout.open(config["DIRNAME"] + "/config.txt");

	for (auto& el : config)
    {
        fout << "\t" << el.first << "\t=\t" << el.second << "\n";
    }
    
    fout.flush();
    fout.close();

}



void save_setup(std::unordered_map<std::string, std::string> config)
{
    boost::filesystem::create_directory(config["DIRNAME"]);

    #ifdef DO_SAVE_WEIGHTS
        boost::filesystem::create_directory(config["DIRNAME"] + "/W1/");
        boost::filesystem::create_directory(config["DIRNAME"] + "/W2/");
    #endif

    std::cout << "saving configuration to:\t" << config["DIRNAME"] << std::endl;
    save_metadata(config);
}






// fout << "COUPLING_HIDDEN =  \t" << HH::a_hidden << "\n";
// fout << "COUPLING_OUT =     \t" << HH::a_out    << "\n";
// fout << "N_LAYER_0 =    \t" << m_n_in       << "\n";
// fout << "N_LAYER_1 =\t" << m_n_hidden   << "\n";
// fout << "N_LAYER_2 =   \t" << m_n_out      << "\n";
// fout << "DELTA_T =      \t" << m_dt         << "\n";
// fout << "SIM_TIME_MS =       \t" << m_sim_time_ms      << "\n";
// fout << "LEARNING_RATE =     \t" << m_learning_rate    << "\n";
// fout << "LATERAL_FACTOR =    \t" << HH::lateral_factor << "\n";
// fout << "mini_batch_size =   \t" << mini_batch_size    << "\n";
