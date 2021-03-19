#include <random>
#include <fstream>

#include "../backprop.h"

// Class used to convert an MNIST image into a poisson distribution.
// Also supports writing this to a file.
class MNISTPoison  
{
private:
    std::vector<std::vector<unsigned char>> const& m_test_images;
    std::vector<std::vector<unsigned char>> const& m_train_images;
    std::vector<std::vector<unsigned>> m_firing_times;
    Vec m_firing_rates;
    int m_max_timesteps, m_max_firings_per, m_total_firings, m_spike_len;

    void GenPoissonSingleCase (int idx,
            std::vector<std::vector<unsigned char>> const& images) 
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        // Generate firing rates based on pixel intensities.
        auto img = images[idx].begin();
        auto it = m_firing_rates.begin();
        while(it != m_firing_rates.end())
            *it++ = (*img++ + 1.0) / 256.0;
        
        // Scale to get firing rates in right range.
        m_firing_rates *= (m_max_firings_per / (double)m_max_timesteps);
        double const expected_total{m_max_timesteps * m_firing_rates.sum()};
        m_firing_rates *= (m_total_firings / (double)expected_total);

        // Use firing_rate to make poisson processes.
        it = m_firing_rates.begin();
        for(int i = 0; i < (int)m_firing_times.size(); ++i)
        {
            auto& spiketrain{m_firing_times[i]};
            spiketrain.resize(0);
            std::exponential_distribution<double> exp(*it++);

            double tstamp{0};
            while(true)
            {
                double const inc{exp(gen)};
                tstamp += inc;

                int const spike_end{std::min<int>(tstamp + m_spike_len, m_max_timesteps)};
                for(; tstamp < spike_end; ++tstamp)
                    spiketrain.push_back((unsigned)tstamp);

                if(spike_end == m_max_timesteps)
                    break;
            }
        }
    }

    void AppendPoisson (std::ofstream& fout) const
    {
        for(auto const& spiketrain: m_firing_times)
        {
            for(unsigned const& spike: spiketrain)
                fout << spike << ',';
            fout << '\n';
        }
    }

public:
    MNISTPoison (int max_timesteps, int max_firings_per, int total_firings, int spike_len,
         std::vector<std::vector<unsigned char>> const& test_images,
         std::vector<std::vector<unsigned char>> const& train_images)
        : m_test_images{test_images},
          m_train_images{train_images},
          m_firing_times(28 * 28, std::vector<unsigned>(max_firings_per)), 
          m_firing_rates(28 * 28),
          m_max_timesteps{max_timesteps}, 
          m_max_firings_per{max_firings_per},
          m_total_firings{total_firings},
          m_spike_len{spike_len}
    {
    }

    void GenerateAndWrite (int const n_train = 60000, int const n_test = 10000)
    {
        for(int idx = 0; idx < n_train; ++idx)
        {
            if(idx % (n_train / 100) == 0)
                std::cout << (int)(100 * idx / (float)n_train) << "%" << std::endl;
            GenPoissonSingleCase(idx, m_train_images);
            std::ofstream fout("mnist/TRAIN_POISSON_" + std::to_string(idx) + ".txt");
            AppendPoisson(fout);
            fout.close();
        }
        for(int idx = 0; idx < n_test; ++idx)
        {
            if(idx % (n_test / 100) == 0)
                std::cout << (int)(100 * idx / (float)n_test) << "%" << std::endl;
            GenPoissonSingleCase(idx, m_test_images);
            std::ofstream fout("mnist/TEST_POISSON_" + std::to_string(idx) + ".txt");
            AppendPoisson(fout);
            fout.close();
        }
    }
};

class PoissonSampler : public SpikeTrainSampler
{
private:
    std::vector<uint8_t> m_labels;
    double m_dt_scalar; // timestep between spikes, in dt units.
    bool m_sample_training; // If true, sample training data. If false, sample test data.

public:
    PoissonSampler(int n_samples, std::vector<uint8_t>&& labels, double dt_scalar, bool sample_training)
        : SpikeTrainSampler(n_samples, !sample_training), 
          m_labels{std::move(labels)}, 
          m_dt_scalar{dt_scalar},
          m_sample_training{sample_training}
    {
    }

    virtual void LoadSpiketrain (int idx, Mat& inp, Vec& desired_out, double dt) override
    {
        (void)dt; // Not used, see note about dt units below.

        // Make desired_out zero everywhere except for label.
        desired_out.setZero();
        desired_out[m_labels[idx]] = 0.4;

        // Open file.
        std::string fl_name{(m_sample_training ? "mnist/TRAIN_POISSON_" : "mnist/TEST_POISSON_")
            + std::to_string(idx) + ".txt"};
        std::ifstream fl(fl_name);
        if(!fl)
            throw std::runtime_error("PoissonSampler file not found: " + fl_name);
        
        // Zero out input matrix.
        inp.setZero();
        char sep;
        unsigned tstamp;
        sep = fl.peek();     
        for(int i = 0; i < 28 * 28; ++i)
        {
            while(sep != '\n' && sep != EOF)
            {
                fl >> tstamp >> sep;
                sep = fl.peek();

                // Note: we ASSUME tstamp is in DT UNITS.
                int const idx{int(m_dt_scalar * tstamp)}; 
                if(idx >= inp.cols())
                {
                    // Out of time range.
                    continue;
                }
                inp(i, idx) += 1;
            }
            fl.ignore();
            sep = fl.peek();
        }
        fl.close();
    }
};
