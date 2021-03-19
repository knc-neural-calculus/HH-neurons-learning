#pragma once

#include <fstream>
#include "network.h"
#include <algorithm>
#include <random>
#include <chrono>

// Abstract class that supports loading a spike train to a matrix. 
struct SpikeTrainSampler 
{
private:
    std::vector<int> m_train_order;
    int m_training_idx;
public:
    SpikeTrainSampler (int n_samples, bool test_sampler) : m_train_order(n_samples), m_training_idx{0} 
    {
        if(test_sampler)
	{
		std::iota(m_train_order.begin(), m_train_order.end(), 0);
	}
	else 
        {
            // Choose random indices by shuffling indices [0, 60000] and picking first max_epochs of them.
            std::vector<int> all_inds(60000);
            std::iota(all_inds.begin(), all_inds.end(), 0);
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(all_inds.begin(), all_inds.end(), 
                         std::default_random_engine(seed));
            std::copy(all_inds.begin(), all_inds.begin() + n_samples, m_train_order.begin());
        }
    }
    virtual void LoadSpiketrain (int const idx, Mat& inp, Vec& desired_out, double dt) = 0;
    void LoadNextSpiketrain (Mat& inp, Vec& desired_out, double dt)
    {
        LoadSpiketrain(m_train_order[m_training_idx++], inp, desired_out, dt);
    }
    void Reset()
    {
        m_training_idx = 0;
    } 
    void ShuffleOrder () 
    {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(m_train_order.begin(), m_train_order.end(), 
                std::default_random_engine(seed));
	Reset();
    } 
};

class BackpropNetwork : public NetworkBase
{
protected:
    // weight and bias derivatives.
    Mat m_W1_velocity, m_W2_velocity; // For momentum SGD
    Float m_gamma = 0.9;
    Mat m_dCdW1, m_dCdW2;
    Mat m_dCdW2_inc;
    std::vector<Mat> m_dCdW1_inc_parts;
    Vec m_dCdb1, m_dCdb2;
    Vec m_desired;
    Float m_cost_scalar;

    std::ofstream loss_out, label_out;
    std::ofstream x1_bias_out, x2_bias_out;

    virtual void DoStuffWithResults () override
    {
        // Backprop increment gradients.
        for(int i = 0; i < m_n_out; ++i)
        {
            Float lhs{m_X2[i].out_deriv};

            // Weight derivatives.
            auto delta_it = m_X2[i].delta.begin();
            for(Float& dCdw2: m_dCdW2_inc.row(i))
                dCdw2 += lhs * (*delta_it++);

            // Bias derivative.
            m_dCdb2[i] += lhs * m_X2[i].bias_deriv;
        }

        for(int i = 0; i < m_n_hidden; ++i)
        {
	    for(int k = 0; k < m_n_out; ++k)
            {
                Float lhs = m_X2[k].out_deriv * m_X2[k].delta_T[i] * m_X1[i].out_deriv;
		auto delta_it = m_X1[i].delta.begin();
                for(Float& dCdw1: m_dCdW1_inc_parts[k].row(i))
                    dCdw1 += lhs * (*delta_it++);
            }
//            // Bias derivative.
//            m_dCdb1[i] += lhs * m_X1[i].bias_deriv;
        }
    }

    Float QuadCost () const
    {
        Float cost = 0;
        for(int i = 0; i < m_n_out; ++i)
            cost += (m_desired[i] > 0.01 ? m_cost_scalar : 1) * (m_out_avg[i] - m_desired[i]) * (m_out_avg[i] - m_desired[i]);
        return 0.5 * cost;
    }

    Float QuadCostPartial (int const i) const 
    {
        return m_out_avg[i] - m_desired[i];
    }

    Float CrossEntropy () const
    {
	Float cost = 0;
	for(int i = 0; i < m_n_out; ++i)
	    if(m_desired[i] > 0.01)
		cost -= m_desired[i] * log(m_out_avg[i]);
            else
		cost -= log(1 - m_out_avg[i]);
	return cost / m_n_out;
    }
 
    Float CrossEntropyPartial (int const i) const
    {
	return m_desired[i] > 0.01 ? -1 / (m_n_out * m_out_avg[i]) 
                                   : 1 / (m_n_out * (1 - m_out_avg[i])); 
    }

    static int LabelFromVec (Vec const& v) 
    {
        return std::max_element(v.begin(), v.end()) - v.begin();
    }

public:
    BackpropNetwork(int n_in, int n_hidden, int n_out, 
        double dt, double sim_time_ms, double learning_rate)
        : NetworkBase(n_in, n_hidden, n_out, dt, sim_time_ms, learning_rate), 
          m_W1_velocity{Mat::Zero(n_hidden, n_in)},
          m_W2_velocity{Mat::Zero(n_out, n_hidden)},
          m_dCdW1{Mat::Zero(n_hidden, n_in)},
          m_dCdW2{Mat::Zero(n_out, n_hidden)},
          m_dCdW1_inc_parts{std::vector<Mat>(n_out, Mat::Zero(n_hidden, n_in))},
          m_dCdW2_inc{Mat::Zero(n_out, n_hidden)},
          m_dCdb1{Vec::Zero(n_hidden)},
          m_dCdb2{Vec::Zero(n_out)},
          m_desired(n_out),
          m_cost_scalar{1}
    {}

    ~BackpropNetwork()
    {
        loss_out.flush();
        loss_out.close();
	label_out.flush();
	label_out.close();
        x1_bias_out.flush();
        x1_bias_out.close();
        x2_bias_out.flush();
        x2_bias_out.close();
    }

    void open_loss_file()
    {
        loss_out.open(dirname + "/loss.txt", std::ios::app);
	label_out.open(dirname + "/labels.txt", std::ios::app);
        x1_bias_out.open(dirname + "/x1_bias.txt", std::ios::app);
        x2_bias_out.open(dirname + "/x2_bias.txt", std::ios::app);
    }

    void SetCostScalar (Float cost_scalar) 
    {
        m_cost_scalar = cost_scalar;
    }

    void LoadInputFromSampler (SpikeTrainSampler* sampler)
    {
        sampler->LoadNextSpiketrain(m_inp, m_desired, m_dt);
    }

    // Stochastic gradient descent.
    // - mini_batch_size Size of batch to be used to calculate average weight partials. 
    // - n_batches Number of mini batches to run for.
    // - sampler SpikeTrainSampler used to get input to network.
    // - same_samples If true, each batch we train on the same sample. Defaults to false.
    // - debug If true, enables debug output. Defaults to false.
    void SGD(int mini_batch_size, int n_batches, SpikeTrainSampler* sampler, 
            bool same_samples = false)
    {
        // Load initial training sample.
        for(int n = 0; n < n_batches; ++n)
        {
            // save weights to file
	    #ifdef DO_SAVE_WEIGHTS
    	        if(n % 100 == 0 )
                        save_weights(n);
    	        else if(n == n_batches - 1)
                        save_weights(n_batches);
	    #endif

            std::cout << "******************************************************BATCH "
                      << n << "*****************************" << std::endl;
    
            // Zero out the partials.
            m_dCdW1.setZero();
            m_dCdW2.setZero();
            m_dCdb1.setZero();
            m_dCdb2.setZero();
    
            double epoch_total_loss = 0;
    
            // Compute averaged partials over mini-batch.
            for(int b = 0; b < mini_batch_size; ++b)
            {
		for(auto& m_dCdW1_inc: m_dCdW1_inc_parts)
                    m_dCdW1_inc.setZero();
                m_dCdW2_inc.setZero();
    
                // Reset the neurons
                std::copy(m_X1_init.begin(), m_X1_init.end(), m_X1.begin());
                std::copy(m_X2_init.begin(), m_X2_init.end(), m_X2.begin());
    
                // Load next training sample.
                try
                {
                    if(same_samples)
                        sampler->LoadSpiketrain(b, m_inp, m_desired, m_dt);
                    else
                        sampler->LoadNextSpiketrain(m_inp, m_desired, m_dt);
                }
                catch(...)
                {
                    return;
                }
    
                // Do training. 
                // This will add the cost partials to the running sums dCdW1, dCdW2.
                FeedForward(true);

		// Incorporate loss partial derivative w/r/t avg. output into backprop calculations.
                for(int i = 0; i < m_n_out; ++i)
                {
                    Float const partial{CrossEntropyPartial(i)};
		    m_dCdW2_inc.row(i) *= partial;
		    m_dCdW1_inc_parts[i] *= partial;
                }

		// Add incremented partials to overal partial estimate.
		m_dCdW2 += m_dCdW2_inc;

		for(auto& m_dCdW1_inc: m_dCdW1_inc_parts)
		    m_dCdW1 += m_dCdW1_inc;

		Float const cost{CrossEntropy()};
                std::cout << "\t";
                Print(m_out_avg);
                std::cout << ',';
                Print(m_desired);
                int const our_label = LabelFromVec(m_out_avg);
                int const real_label = LabelFromVec(m_desired);
                std::cout << ", " << (our_label == real_label ? "true" : "false")
                          << ", " << cost << std::endl;
                
                loss_out << cost << ",";
		label_out << real_label << ",";

                epoch_total_loss += cost;
            }
    
            std::cout << "avg epoch loss:\t" << epoch_total_loss / mini_batch_size << std::endl;
    
            loss_out << "\n";
            loss_out.flush();

            label_out << "\n";
            label_out.flush();
    
            // Convert sums to averages.
            m_dCdW1 /= mini_batch_size;
            m_dCdW2 /= mini_batch_size;
            m_dCdb1 /= m_sim_incs * mini_batch_size;
            m_dCdb2 /= m_sim_incs * mini_batch_size;
    
            // Gradient descent update using momentum.
            m_W1_velocity = m_gamma * m_W1_velocity + m_learning_rate * m_dCdW1;
            m_W1 = m_W1 - m_W1_velocity;
            m_W2_velocity = m_gamma * m_W2_velocity + m_learning_rate * m_dCdW2;
            m_W2 = m_W2 - m_W2_velocity;
        }
    }

    double EvaluateNetwork(SpikeTrainSampler* sampler, int const eval_idx, int const n_test_samples=500)
    {
        // Load initial training sample.
        int n_total{0}, n_hit{0};
        std::ofstream percent_out(dirname + "/percent" + std::to_string(eval_idx) + ".txt");

        double percent_hit;
        for(; n_total < n_test_samples; ++n_total)
        {
            // Reset the neurons
            std::copy(m_X1_init.begin(), m_X1_init.end(), m_X1.begin());
            std::copy(m_X2_init.begin(), m_X2_init.end(), m_X2.begin());

            // Evaluate network. 
            FeedForward(false);
            int const our_label = LabelFromVec(m_out_avg);
            int const real_label = LabelFromVec(m_desired);
            if(our_label == real_label)
                ++n_hit;

            // Load next training sample.
            try
            {
                sampler->LoadNextSpiketrain(m_inp, m_desired, m_dt);
            }
            catch(...)
            {
                break; // Done with all the test data
            }
            percent_hit = n_hit / (double)(n_total+1) * 100;
            percent_out << (n_total+1) << "\t" << percent_hit << "\n";
        }
        // Write percentage.
        percent_out.close();
        return percent_hit;
    }
};
