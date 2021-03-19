#pragma once

#include <ctime>
#include <fstream>
#include <random>

#include "neuron_models/hh.h"
#include "neuron_models/ml.h"
#include "neuron_models/san.h"

typedef HH Neuron;

void Print (Vec const& v)
{
    std::cout << '[';
    for(auto& val: v)
        std::cout << val << ',';
    std::cout << ']';
}

class NetworkBase 
{
protected:
    int m_n_in, m_n_hidden, m_n_out;
    double m_dt; // Timestep in ms.

    // This is the number of timesteps used in a single spiketrain input.
    // We use this to preallocate the spiketrain matrices. 
    int m_sim_incs;
    double m_sim_time_ms; // Simulation time in ms.

    double m_learning_rate;

    // Spiketrain matrices:
    // ROWS of matrix are the individual spike trains. COLUMNS give the spike train 
    // input at a certain timestamp to all the neurons.
    
    // Used to store (un)weighted input spiketrains.
    Mat m_inp, m_winp;
    
    // Vectors of outputs from each layer.
    Vec m_X1_out, m_X2_out;

    Vec m_out_avg;

    // Network neurons, weight matrices.
    std::vector<Neuron> m_X1, m_X2;
    std::vector<Neuron> m_X1_init, m_X2_init;
    Mat m_W1, m_W2;

    bool m_debug_print_voltages = false;
    std::ofstream m_X1_V_out, m_X2_V_out;

public:
    // location of the configuration data and weights
    std::string dirname;


    // Called every timestep in FeedForward.
    virtual void DoStuffWithResults () 
    {
    }

public:
    NetworkBase(int n_in, int n_hidden, int n_out, double dt, double sim_time_ms, double learning_rate, bool do_transient=true)
        : m_n_in{n_in}, m_n_hidden{n_hidden}, m_n_out{n_out}, m_dt{dt},
          m_sim_incs{int(sim_time_ms / dt)}, m_sim_time_ms{sim_time_ms},
          m_learning_rate{learning_rate},
          m_inp{Mat::Zero(n_in, m_sim_incs)}, 
          m_winp{Mat(n_hidden, m_sim_incs)},
          m_X1_out(n_hidden), m_X2_out(n_out),
          m_out_avg(n_out)
    {

        // Note that output neurons have more params because we need to store more 
        // partials for backprop on the first layer of weights.
        m_X1 = std::vector<Neuron>(n_hidden, Neuron(n_in, m_dt, false));
        m_X2 = std::vector<Neuron>(n_out, Neuron(n_hidden, m_dt, true));

        // Normal distribution weights.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<Float> norm_dist(0.5, 0.17);
        m_W1 = Mat::Zero(n_hidden, n_in);
        for(Float& val: m_W1.reshaped())
            val = norm_dist(gen);

        m_W2 = Mat::Zero(n_out, n_hidden);
        for(Float& val: m_W2.reshaped())
            val = norm_dist(gen);

        // (0,1)
        // m_W1 = Mat::Random(n_hidden, n_in).cwiseAbs();
        // m_W2 = Mat::Random(n_out, n_hidden).cwiseAbs();
        
        // Transient setup to get to a more realistic state.
        // We just pass all ones in as the network input.
        if(do_transient)
             FeedForward(true);
        m_X1_init = m_X1;
        m_X2_init = m_X2;
    }

    // Does a simulation for m_sim_incs * m_dt milliseconds.
    void FeedForward (bool training_case)
    {
        m_out_avg.setZero();

        // Compute weighted input.
        m_winp.noalias() = m_W1 * m_inp;

        for(int t = 0; t < m_sim_incs; ++t)
        {
            // Simulate hidden layer neurons.

            static const Vec blank(0);
            auto winp_it = m_winp.col(t).begin();
            auto X1_out_it = m_X1_out.begin();
	    Float out_sum = 0;
            for (Neuron& x1 : m_X1)
            {
                *X1_out_it++ = x1.Integrate(*winp_it++, m_inp.col(t), blank, training_case);
		out_sum += x1.V > Neuron::Vt;
            }
            
            for (Neuron& x1: m_X1)
            {
		if(x1.V <= Neuron::Vt) 
                {
			x1.V = x1.V * pow(
			    (Float) ( m_n_hidden - out_sum ) / (Float) m_n_hidden,
			    Neuron::LF_hidden
			);
                }
            }
            
            // Compute weighted input to output layer neurons.
            // We just use m_X2_out as a temporary vector to store this.
            m_X2_out.noalias() = m_W2 * m_X1_out;            

            // Simulate output layer neurons.
            auto W2_row_it = m_W2.rowwise().begin();
            auto X2_out_it = m_X2_out.begin();
	    out_sum = 0;
            for (Neuron& x2 : m_X2)
            {
                *X2_out_it++ = x2.Integrate(*X2_out_it, m_X1_out, *W2_row_it++, training_case);
		out_sum += x2.V > Neuron::Vt;
            }

            for (Neuron& x2 : m_X2)
            {
		if(x2.V <= Neuron::Vt) 
                {
			x2.V = x2.V * pow(
			    (Float) ( m_n_out - out_sum ) / (Float) m_n_out,
			    Neuron::LF_out
			);
                }
            }

            // Add to output average
            m_out_avg += m_X2_out;

            // Let someone do something with our shit.
            if(training_case)
                DoStuffWithResults();

            if(m_debug_print_voltages)
            {
                for(Neuron const& n: m_X1)
                    m_X1_V_out << n.V << ',';
                m_X1_V_out << std::endl;

                for(Neuron const& n: m_X2)
                    m_X2_V_out << n.V << ',';
                m_X2_V_out << std::endl;
            }
        }
        m_out_avg /= m_sim_incs;
    }

    Vec GetOutputAverage () const
    {
	return m_out_avg;
    }

    // Gets a reference to the ith input spike train, to be modified. 
    Mat::RowXpr GetInputHandle (int i)
    {
        // Note: Row pointer might be less efficient. We should think about this.
        return m_inp.row(i);
    }

    void mat_print_weights(Mat m, std::string filename)
    {
        std::ofstream fout;

        fout.open(filename);
        for (int i = 0; i < m.rows(); i++)
        {
            for (int j = 0; j < m.cols(); j++)
            {
                fout << m(i,j) << ",";
            }
            fout << "\n";
        }
        fout.flush();
        fout.close();
    }
        
        
    inline void save_weights(int const n)
    {
        mat_print_weights(m_W1, dirname + "/W1/weights_W1_e-" + std::to_string(n) + ".csv");
        mat_print_weights(m_W2, dirname + "/W2/weights_W2_e-" + std::to_string(n) + ".csv");
    }

    void ReadWeightsAndBiases (std::string inp_dirname, int const epoch)
    {
        std::ifstream f_W1(inp_dirname + "/W1/weights_W1_e-" + std::to_string(epoch) + ".csv");
        std::ifstream f_W2(inp_dirname + "/W2/weights_W2_e-" + std::to_string(epoch) + ".csv");
        std::ifstream f_X1_biases(inp_dirname + "/x1_bias.txt");
        std::ifstream f_X2_biases(inp_dirname + "/x2_bias.txt");

        if (!f_W1 || !f_W2 || !f_X1_biases || !f_X2_biases)
        {
            throw std::runtime_error("failed to open one of weights or biases files for reading");
        }

        char comma;
        for(auto row: m_W1.rowwise())
        {
            for(Float& el: row)
                f_W1 >> el >> comma;
            f_W1.ignore(); // Skip \n
        }
        for(auto row: m_W2.rowwise())
        {
            for(Float& el: row)
                f_W2 >> el >> comma;
            f_W2.ignore(); // Skip \n
        }
        for(int e = 0; e < epoch; ++e)
        {
            for(int i = 0; i < m_n_hidden; ++i)
                f_X1_biases >> m_X1[i].bias >> comma;
            f_X1_biases.ignore(); // Skip \n. If we hit EOF, f_W2 becomes false.
        }
        for(int e = 0; e < epoch; ++e)
        {
            for(int i = 0; i < m_n_out; ++i)
                f_X2_biases >> m_X2[i].bias >> comma;
            f_X2_biases.ignore(); // Skip \n. If we hit EOF, f_W2 becomes false.
        }
        f_W1.close();
        f_W2.close();
        f_X1_biases.close();
        f_X2_biases.close();
    }

    void ToggleVoltagePrinting (bool val)
    {
        m_debug_print_voltages = val;
        if(val)
        {
            m_X1_V_out.open(dirname + "/X1_voltages.csv");
            m_X2_V_out.open(dirname + "/X2_voltages.csv");
        }
    }

    ~NetworkBase()
    {
        m_X1_V_out.close();
        m_X2_V_out.close();
    }
};
