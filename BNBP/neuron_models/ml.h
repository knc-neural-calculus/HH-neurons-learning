#pragma once 
#include "defines.h"
#include <math.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cmath>

// Define stuff related to Morris-Lecar model.

class ML
{
public:
    static constexpr float gca = 4.0;
    static constexpr float gk = 8;
    static constexpr float gl = 2;

    static constexpr float Eca = 120;
    static constexpr float Ek = -84;
    static constexpr float El = -60;

    static constexpr float Vt = 20.0;
    static constexpr float Kp = 1.0;

    static constexpr float V_tune_1 = -1.2;
    static constexpr float V_tune_2 = 18;
    static constexpr float V_tune_3 = 12;
    static constexpr float V_tune_4 = 17.4;
    static constexpr float phi = 1/15.0;
    static constexpr float C = 20;

//======================NEURON=PARAMETERS=================================================
    #ifdef NEURON_ARE_VARS
        static Float a_hidden;
        static Float a_out;
        static Float LF_hidden;
        static Float LF_out;
    #else
        static constexpr Float a_hidden = 100;
        static constexpr Float a_out = 100;
        static constexpr Float LF_hidden = 0.5;
        static constexpr Float LF_out = 0.5;
    #endif
//=======================================================================================

    Float V, W;
    Float output; // Neurotransmitter output from neuron.
    Float out_deriv; // Derivative of output function.
    double const dt;
    int const n_upstream;
    Vec delta, delta_T;
    Float bias_deriv;
    Float coup_scalar;
    Float bias;

    ML& operator= (ML const& rhs)
    {
        V = rhs.V;
        W = rhs.W;
        std::copy(rhs.delta.begin(), rhs.delta.end(), delta.begin());
        std::copy(rhs.delta_T.begin(), rhs.delta_T.end(), delta_T.begin());
        bias_deriv = rhs.bias_deriv;
        coup_scalar = rhs.coup_scalar;
        // NOTE: bias not set here, since we don't want to overwrite it when we reset the neurons.
        return *this;
    }

    ML (int n_upstream_, double dt_, bool is_output)
        : V{(Float)-52.14}, W{(Float)0.02}, 
        output{(Float)0}, out_deriv{(Float)0},
        dt{dt_}, n_upstream{n_upstream_},
        delta{Vec::Zero(n_upstream)},
        delta_T{is_output ? delta : Vec(0)},
        bias_deriv{0},
        coup_scalar{is_output ? a_out : a_hidden},
        bias{0}
    {
    }

    // Performs Euler integration.
    //   - winp : Weighted input to this neuron.
    //   - T : Unweighted output of previous neuron layer.
    //   - W_in : Row of previous weight matrix associated with this neuron.
    //   - training_case : If true, we will simulate deltas for learning.
    //   - Return : Output neurotransmitter.
    Float Integrate(Float const winp, Vec const& T, Vec const& W_in, bool training_case) 
    {
        Float sclr{coup_scalar / n_upstream};
        Float I{(Float)40.5f + sclr * winp + bias}; // Note we add 30.5 since the bifurcation occurs at 40.5 in ML.

        Float const Mss = 1/(1 + exp(-2*(V - V_tune_1)/V_tune_2));
        Float const Wss = 1/(1 + exp(-2*(V - V_tune_3)/V_tune_4));
        Float const Tw = 1/(phi*cosh((V - V_tune_3)/(2*V_tune_4)));

        Float const G = gca * Mss + gk * W + gl;
        Float const E = gca * Mss * Eca + gk * W * Ek + gl * El;

        // Euler's method. Since the input parameters are constant, using RK4 would 
        // still be first order.
        V += dt * (I - G * V + E) / C;
        W += dt * (Wss - W) / Tw;

        // Compute output and derivative.
        Float const val = exp(-(V - Vt) / Kp);
        output = 1 / (1 + val);
        out_deriv = val / (Kp * (val+1) * (val+1));

        if(training_case)
        {
            // Integrate vector (dVdw)_i for all upstream neuron connections i.
            // NOTE: WE USE EULER'S METHOD HERE SINCE LEAPFROG DOESN'T SEEM TO DO WELL!
            delta += dt * (T * Float(sclr) - G * delta) / C;
            
            // Bias derivative.
            bias_deriv += dt * (1 - G * bias_deriv) / C;

            // Integrate vector (dVdT)_i for all downstream neuron connections i. This is 
            // only needed for the output layer in backprop.
            if(delta_T.size() > 0)
                delta_T += dt * (W_in * Float(sclr) - G * delta_T) / C;
        }

        return output;
    }
};

#ifdef NEURON_ARE_VARS
Float ML::a_hidden = 0;
Float ML::a_out = 0;
Float ML::LF_hidden = 0;
Float ML::LF_out = 0;
#endif

