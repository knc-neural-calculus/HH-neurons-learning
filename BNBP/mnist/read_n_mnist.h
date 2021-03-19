#include <string>
#include <vector>

#include "../backprop.h"

// I found this to be the max size of one of the event streams.
#define __N_MNIST_ALLOC_MAX__ 9000
typedef std::array<unsigned char, 5> BinaryNMNISTEvent; 
typedef std::array<BinaryNMNISTEvent, __N_MNIST_ALLOC_MAX__> EventStream;

class NMNISTReader : public SpikeTrainSampler
{
private:
    std::string m_folder;
    std::unique_ptr<EventStream> m_event_stream;
    std::vector<uint8_t> m_training_labels;

public:
    NMNISTReader(int n_samples, std::string folder, std::vector<uint8_t>&& training_labels) 
        : SpikeTrainSampler(n_samples, false),
          m_folder{folder}, 
          m_event_stream{new EventStream}, m_training_labels{std::move(training_labels)}
    {
        if(m_folder.back() != '/' && m_folder.back() != '\\')
            m_folder += '/';
    }

    virtual void LoadSpiketrain (int const idx, Mat& inp, Vec& desired_out, double dt) override
    {
        // Make desired_out zero everywhere except for label.
        desired_out.setZero();
        desired_out[m_training_labels[idx]] = 1.0;

        // Construct file name by padding index with zeros.
        std::string file_name{std::to_string(idx + 1)};
        file_name = m_folder + std::string(5 - file_name.size(), '0') + file_name + ".bin";

        // Use FILE*, not fstream because I am a chad (also way faster).
        FILE* fl{fopen(file_name.c_str(), "rb")};
        if(!fl)
            throw std::runtime_error("NMNISTReader file not found: " + file_name);

        // Read in one call.
        fread(m_event_stream->data(), 5, __N_MNIST_ALLOC_MAX__, fl);

        // Get number of entries.
        int n_entries = (int)ftell(fl) / 5;
        
        // Zero out input matrix.
        inp.setZero();
        
        // Convert event stream to matrix of inputs.
        static const int all_but_last_mask = (1 << 7) - 1;
        for(int i = 0; i < n_entries; ++i)
        {
            // Each event occupies 40 bits as described below:
            // bit 39 - 32: Xaddress (in pixels)
            // bit 31 - 24: Yaddress (in pixels)
            // bit 23: Polarity (0 for OFF, 1 for ON)
            // bit 22 - 0: Timestamp (in microseconds)
            auto const& event{(*m_event_stream)[i]};
            int const x = event[0], y = event[1];

            // Some samples lie outside the region [0, 28] in the N-MNIST dataset. 
            // Not sure why... Just discard them.
            if(x >= 28 || y >= 28)
                continue;

            int timestamp = all_but_last_mask & event[2];
            timestamp = (timestamp << 8) | event[3];
            timestamp = (timestamp << 8) | event[4];

            // HACK : SPEED UP INPUT 10 TIMES 
            // TODO : IS THIS NECESSARY ???? 
            timestamp /= 10;

            // Get interval in which timestamp occurs. 
            // Timestamp is in microseconds.
            int idx = (int)((timestamp * 0.001) / dt);
            if(idx >= inp.cols())
            {
//                Float timestamp_ms = timestamp * 0.001;
//                std::cout << 
//                    " WARNING: N-MNIST Sample with timestamp " 
//                    + std::to_string(timestamp_ms) 
//                    + " doesn't fit within simulation timeframe " 
//                    + std::to_string(dt * inp.cols()) 
//               << std::endl;
                continue;
            }
            inp(y * 28 + x, idx) = 1;
            if(idx + 2 < inp.cols())
            {
                inp(y * 28 + x, idx + 1) = 1;
                inp(y * 28 + x, idx + 2) = 1;
            }
        }
        fclose(fl);
    }
};
