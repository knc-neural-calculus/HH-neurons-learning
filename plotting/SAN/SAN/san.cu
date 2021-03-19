#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <fstream>
#include <stdio.h>
#include <vector>

struct compute_san_output
{
    static constexpr float an_coupling_threshold = -60;
    static constexpr float Tmax = 1.0;
    static constexpr float C = 1;
    static constexpr float A = 0.02;
    static constexpr float vL = -60.95;
    static constexpr float vNa = 55;
    static constexpr float vK = -100;
    static constexpr float tauhA = 15;
    static constexpr float vCa = 120;
    static constexpr float kD = 30;
    static constexpr float vAMPA = 0;
    static constexpr float vNMDA = 0;
    static constexpr float vGABA = -70;
    static constexpr float alphaCa = 0.5;
    static constexpr float tauAMPA = 2;
    static constexpr float tausNMDA = 100;
    static constexpr float tauxNMDA = 2;
    static constexpr float tauGABA = 10;
    static constexpr float gL = 0.016307;
    static constexpr float gNa = 12.2438;
    static constexpr float gK = 19.20436;
    static constexpr float gA = 1.79259;
    static constexpr float gKS = 0.0350135;
    static constexpr float gNaP = 0.63314;
    static constexpr float gAR = 0.0166454;
    static constexpr float gCa = 0.1624;
    static constexpr float gKCa = 0.7506;
    static constexpr float gAMPA = 0.513425;
    static constexpr float gNMDA = 0.00434132;
    static constexpr float gGABA = 0.00252916;
    static constexpr float tauCa = 739.09;

    static constexpr float dt = 0.01;
    static constexpr float n_incs = 40000;

    __host__ __device__
        float operator()(float T)
    {
        // State variables.
        float V = 0.0, Ca = 0.1, nK = 0.0;

        // Intermediate variables.
        float mCa_inf, mKCa_inf, mNaP_inf,
            alpha_n, beta_n, ICa,
            dnKdt, dCadt, dvdt;

        float avg_out = 0.0;
        for (int i = 0; i < n_incs; ++i)
        {
            mCa_inf = 1 / (1 + expf(-(V + 20) / 9));
            mKCa_inf = 1 / (1 + powf(kD / Ca, 3.5));
            mNaP_inf = 1 / (1 + expf(-(V + 55.7) / 7.7));

            alpha_n = V == -34 ? 0.1 : 0.01 * (V + 34) / (1 - expf(-(V + 34) / 10));
            beta_n = 0.125 * expf(-(V + 44) / 25);

            // Calculate gating variable derivatives
            dnKdt = 4 * (alpha_n * (1 - nK) - beta_n * nK);

            // Calculate Ca2+ derivative
            ICa = gCa * mCa_inf * mCa_inf * (V - vCa);
            dCadt = -alphaCa * (10 * A * ICa) - Ca / tauCa;

            // Calculate voltage derivative
            dvdt = T
                - gK * powf(nK, 4) * (V - vK)
                - ICa
                - gKCa * mKCa_inf * (V - vK)
                - gNaP * mNaP_inf * mNaP_inf * mNaP_inf * (V - vNa)
                - gL * (V - vL);

            V += dt * dvdt;
            Ca += dt * dCadt;
            nK += dt * dnKdt;
            avg_out += 1 / (1 + expf(-(V + 55)/2.275));
        }
        return avg_out / n_incs;
    }

    static void record(float T, int n_incs_record, std::ostream& out)
    {
        // State variables.
        float V = 0.0, Ca = 0.1, nK = 0.0;

        // Intermediate variables.
        float mCa_inf, mKCa_inf, mNaP_inf,
            alpha_n, beta_n, ICa,
            dnKdt, dCadt, dvdt;

        for (int i = 0; i < n_incs_record; ++i)
        {
            mCa_inf = 1 / (1 + expf(-(V + 20) / 9));
            mKCa_inf = 1 / (1 + powf(kD / Ca, 3.5));
            mNaP_inf = 1 / (1 + expf(-(V + 55.7) / 7.7));

            alpha_n = V == -34 ? 0.1 : 0.01 * (V + 34) / (1 - expf(-(V + 34) / 10));
            beta_n = 0.125 * expf(-(V + 44) / 25);

            // Calculate gating variable derivatives
            dnKdt = 4 * (alpha_n * (1 - nK) - beta_n * nK);

            // Calculate Ca2+ derivative
            ICa = gCa * mCa_inf * mCa_inf * (V - vCa);
            dCadt = -alphaCa * (10 * A * ICa) - Ca / tauCa;

            // Calculate voltage derivative
            dvdt = T
                - gK * powf(nK, 4) * (V - vK)
                - ICa
                - gKCa * mKCa_inf * (V - vK)
                - gNaP * mNaP_inf * mNaP_inf * mNaP_inf * (V - vNa)
                - gL * (V - vL);

            V += dt * dvdt;
            Ca += dt * dCadt;
            nK += dt * dnKdt;
            out << dt * i << ',' << V << ',' << Ca << ',' << nK << std::endl;
        }
    }
};

struct compute_hh_output
{
    constexpr static const float gna = 120;  constexpr static const float ena = 55;                      // Sodium conductance and potential
    constexpr static const float gk = 36; constexpr static const float ek = -72;                         // Potassium conductance and potential
    constexpr static const float gl = 0.3; constexpr static const float el = -50;                        // Leak conductance and potential
    static constexpr float dt = 0.01;
    static constexpr float n_incs = 40000;

    __host__ __device__
        float operator()(float T)
    {
        // State variables.
        float V = 0.0, M = 0.0, N = 0.0, H = 1.0;

        // Intermediate variables.
        float Am, An, Ah, Bm, Bn, Bh,
              dvdt, dMdt, dNdt, dHdt;

        float avg_out = 0.0;
        for (int i = 0; i < n_incs; ++i)
        {
            // Calculate intermediate quantities
            Am = (3.5 + 0.1 * V) / (1 - exp(-3.5 - 0.1*V));
            An = (-0.5 - 0.01 * V) / (exp(-5 - 0.1 * V) - 1);
            Ah = 0.07 * exp(-V / 20 - 3);

            Bm = 4 * exp(-(V + 60) / 18);
            Bn = 0.125*exp(-(V + 60) / 80);
            Bh = 1 / (exp(-3 - 0.1*V) + 1);

            // Calculate gating variable derivatives
            dMdt = Am * (1 - M) - Bm * M;
            dNdt = An * (1 - N) - Bn * N;
            dHdt = Ah * (1 - H) - Bh * H;

            // Calculate voltage derivative
            dvdt = T
                - gna * M * M * M * H * (V - ena)
                - gk * powf(N, 4) * (V - ek)
                - gl * (V - el);

            V += dt * dvdt;
            M += dt * dMdt;
            N += dt * dNdt;
            H += dt * dHdt;
            avg_out += 1 / (1 + expf(-(V + 20)/3));
        }
        return avg_out / n_incs;
    }
};

int main()
{
    std::ofstream san_sample("san_sample.csv");
    compute_san_output::record(1, 2000 * 100, san_sample);
    san_sample.close();

    std::ofstream san_iv_plot("san_iv_plot_no_zoom.csv");
    int N = 1e6;
    thrust::device_vector<float> output(N), current(N);
    thrust::sequence(current.begin(), current.end(), 0.0f, 5.0f / N);
    thrust::transform(current.begin(), current.end(), output.begin(), compute_san_output());
    thrust::host_vector<float> output_cpu(output), current_cpu(current);
    for (int i = 0; i < N; ++i)
        san_iv_plot << current_cpu[i] << ',' << output_cpu[i] << std::endl;
    san_iv_plot.close();

    san_iv_plot.open("san_iv_plot_zoom_1.csv");
    thrust::sequence(current.begin(), current.end(), 2.4f, 0.7f / N);
    thrust::transform(current.begin(), current.end(), output.begin(), compute_san_output());
    output_cpu = output;
    current_cpu = current;
    for (int i = 0; i < N; ++i)
        san_iv_plot << current_cpu[i] << ',' << output_cpu[i] << std::endl;
    san_iv_plot.close();

    san_iv_plot.open("san_iv_plot_zoom_2.csv");
    thrust::sequence(current.begin(), current.end(), 2.7f, 0.1f / N);
    thrust::transform(current.begin(), current.end(), output.begin(), compute_san_output());
    output_cpu = output;
    current_cpu = current;
    for (int i = 0; i < N; ++i)
        san_iv_plot << current_cpu[i] << ',' << output_cpu[i] << std::endl;
    san_iv_plot.close();

    std::ofstream hh_iv_plot("hh_iv_plot.csv");
    thrust::sequence(current.begin(), current.end(), 5.0f, 10.0f / N);
    thrust::transform(current.begin(), current.end(), output.begin(), compute_hh_output());
    output_cpu = output;
    current_cpu = current;
    for (int i = 0; i < N; ++i)
        hh_iv_plot << current_cpu[i] << ',' << output_cpu[i] << std::endl;
    hh_iv_plot.close();
}
