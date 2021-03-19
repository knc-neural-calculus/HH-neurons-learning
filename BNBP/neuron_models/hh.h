#pragma once 
#include "defines.h"
#include <math.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cmath>

// Define stuff related to Hodgkin Huxley model.

class HH
{
public:

//======================NEURON=PARAMETERS=================================================

    static constexpr Float Vt = 20.0;
    static constexpr Float Kp = 3.0;

    #ifdef NEURON_ARE_VARS
        static Float gna; 
        static Float gk;
        static Float gl;

        static Float Ena;
        static Float Ek;
        static Float El;
        static Float a_hidden;
        static Float a_out;
        static Float LF_hidden;
        static Float LF_out;
        static Float use_bias;
        static Float use_beta_phase_2;
    #else
        static constexpr Float gna = 120.0; 
        static constexpr Float gk = 36.0;
        static constexpr Float gl = 0.3;

        static constexpr Float Ena = 115.0;
        static constexpr Float Ek = -12.0;
        static constexpr Float El = 10.613;
        static constexpr Float a_hidden = 100;
        static constexpr Float a_out = 100;
        static constexpr Float LF_hidden = 0.5;
        static constexpr Float LF_out = 0.5;
        static constexpr Float use_bias = 0.0;
        static constexpr Float use_beta_phase_2 = 0.0;
    #endif
//=======================================================================================


    // static float lerp (Float const a, Float const b, Float const t) 
    // {
    //     return a + t * (b - a);
    // }

    Float V, M, N, H;
    Float output; // Neurotransmitter output from neuron.
    Float out_deriv; // Derivative of output function.
    double const dt;
    int const n_upstream;
    Vec delta, delta_T;
    Float bias_deriv;
    Float coup_scalar;
    Float bias;

    HH& operator= (HH const& rhs)
    {
        V = rhs.V;
        M = rhs.M;
        N = rhs.N;
        H = rhs.H;
        std::copy(rhs.delta.begin(), rhs.delta.end(), delta.begin());
        std::copy(rhs.delta_T.begin(), rhs.delta_T.end(), delta_T.begin());
        bias_deriv = rhs.bias_deriv;
        coup_scalar = rhs.coup_scalar;
        // NOTE: bias not set here, since we don't want to overwrite it when we reset the neurons.
        return *this;
    }

    HH (int n_upstream_, double dt_, bool is_output)
        : V{(Float)0}, M{(Float)0}, N{(Float)0}, H{(Float)1}, 
        output{(Float)0}, out_deriv{(Float)0},
        dt{dt_}, n_upstream{n_upstream_},
        delta{Vec::Zero(n_upstream)},
        delta_T{is_output ? delta : Vec(0)},
        bias_deriv{0},
        coup_scalar{is_output ? a_out : a_hidden},
        bias{0}
    {
    }

    // Performs leapfrog integration.
    //   - winp : Weighted input to this neuron.
    //   - T : Unweighted output of previous neuron layer.
    //   - W_in : Row of previous weight matrix associated with this neuron.
    //   - training_case : If true, we will simulate deltas for learning.
    //   - Return : Output neurotransmitter.
    Float Integrate(Float const winp, Vec const& T, Vec const& W_in, bool training_case) 
    {
        // Intermediate quantities.
        Float const aH = 0.07*exp(-V / 20.0);
        Float const aM = V == 25.0 ? 1 :
                   (25.0 - V) / (10.0*(exp((25.0 - V) / 10.0) - 1.0));

        Float const aN = V == 10.0 ? 1 : 
                   (10.0 - V) / (100.0*(exp((10.0 - V) / 10.0) - 1.0));
        
        Float const bH = 1.0 / (exp((30.0 - V) / 10.0) + 1.0);
        Float const bM = 4.0*exp(-V / 18.0);
        Float const bN = use_beta_phase_2 > 0.5 ? 0.125 * exp(-(V + 70) / 19.7) : 0.125*exp(-V / 80.0);

        // Gating variable update.
        M = (aM * dt + (1 - dt/2 * (aM + bM)) * M) / (dt/2 * (aM + bM) + 1);
        N = (aN * dt + (1 - dt/2 * (aN + bN)) * N) / (dt/2 * (aN + bN) + 1);
        H = (aH * dt + (1 - dt/2 * (aH + bH)) * H) / (dt/2 * (aH + bH) + 1);

        Float const G = gna*M*M*M*H + gk*pow(N, 4) + gl;
        Float const E = gna*M*M*M*H*Ena + gk*pow(N, 4)*Ek + gl*El;

        // This quantity is used for updating V, delta and delta_T.
        Float const rhs_half_timestep = (1 - G * dt/2) / (1 + dt/2 * G);
        
        Float sclr{coup_scalar / n_upstream};

        // Voltage update.
        V = dt * ((use_bias * bias) + winp*sclr + E) / (1+dt/2 * G) + V * rhs_half_timestep;

        // Compute output and derivative.
        Float const val = exp(-(V - Vt) / Kp);
        output = 1 / (1 + val);
        out_deriv = val / (Kp * (val+1) * (val+1));

        if(training_case)
        {
            // Integrate vector (dVdw)_i for all upstream neuron connections i.
            // NOTE: WE USE EULER'S METHOD HERE SINCE LEAPFROG DOESN'T SEEM TO DO WELL!
            delta += dt * (T * Float(sclr) - G * delta);
            

            // Bias derivative.
            bias_deriv += dt * (1 - G * bias_deriv);

            // Integrate vector (dVdT)_i for all downstream neuron connections i. This is 
            // only needed for the output layer in backprop.
            if(delta_T.size() > 0)
                delta_T += dt * (W_in * Float(sclr) - G * delta_T);
        }

        return output;
    }
};

#ifdef NEURON_ARE_VARS
Float HH::gna = 120.0; 
Float HH::gk = 36.0;
Float HH::gl = 0.3;
Float HH::Ena = 115.0;
Float HH::Ek = -12.0;
Float HH::El = 10.613;
Float HH::a_hidden = 0;
Float HH::a_out = 0;
Float HH::LF_hidden = 0;
Float HH::LF_out = 0;
Float HH::use_bias = 0.0;
Float HH::use_beta_phase_2 = 0.0;
#endif
