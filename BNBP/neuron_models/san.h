#pragma once 
#include "defines.h"
#include <math.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cmath>

// Define stuff related to Hodgkin Huxley model.

class SAN
{
public:

//======================NEURON=PARAMETERS=================================================
//
    static constexpr Float Vt = -35.0;
    static constexpr Float Kp = 3.6;

    #ifdef NEURON_ARE_VARS
	static Float Ar;
	static Float Ad;
	static Float g_coupling;
	static Float e_coupling_gaba;
	
	static Float Tmax;
	static Float A;
	static Float vL;
	static Float vNa;
	static Float vK;
	static Float tauhA;
	static Float vCa;
	static Float kD;
	static Float vAMPA;
	static Float vNMDA;
	static Float vGABA;
	static Float alphaCa;
	static Float tauAMPA;
	static Float tausNMDA;
	static Float tauxNMDA;
	static Float tauGABA;
	static Float gL;
	static Float gNa;
	static Float gK;
	static Float gA;
	static Float gKS;
	static Float gNaP;
	static Float gAR;
	static Float gCa;
	static Float gKCa;
	static Float gAMPA;
	static Float gNMDA;
	static Float gGABA;
	static Float tauCa;
        static Float a_hidden;
        static Float a_out;
	static Float LF_hidden;
	static Float LF_out;
	static Float use_bias;
    #else 
	static constexpr Float Ar = 5.0 
	static constexpr Float Ad = 0.18
	static constexpr Float g_coupling = 0.01
	static constexpr Float e_coupling_gaba = -70.0
	
	static constexpr Float Tmax = 1.0;
	static constexpr Float A = 0.02;
	static constexpr Float vL = -60.95;
	static constexpr Float vNa = 55;
	static constexpr Float vK = -100;
	static constexpr Float tauhA = 15;
	static constexpr Float vCa = 120;
	static constexpr Float kD = 30;
	static constexpr Float vAMPA = 0;
	static constexpr Float vNMDA = 0;
	static constexpr Float vGABA = -70;
	static constexpr Float alphaCa = 0.5;
	static constexpr Float tauAMPA = 2;
	static constexpr Float tausNMDA = 100;
	static constexpr Float tauxNMDA = 2;
	static constexpr Float tauGABA = 10;
	static constexpr Float gL = 0.016307;
	static constexpr Float gNa = 12.2438;
	static constexpr Float gK = 19.20436;
	static constexpr Float gA = 1.79259;
	static constexpr Float gKS = 0.0350135;
	static constexpr Float gNaP = 0.63314;
	static constexpr Float gAR = 0.0166454;
	static constexpr Float gCa = 0.1624;
	static constexpr Float gKCa = 0.7506;
	static constexpr Float gAMPA = 0.513425;
	static constexpr Float gNMDA = 0.00434132;
	static constexpr Float gGABA = 0.00252916;
	static constexpr Float tauCa = 739.09;
        static constexpr Float a_hidden = 0;
        static constexpr Float a_out = 0;
	static constexpr Float LF_hidden = 0;
	static constexpr Float LF_out = 0;
	static constexpr Float use_bias = 0.0;
    #endif 
//=======================================================================================

    Float V, Nk, Ca;
    Float output; // Neurotransmitter output from neuron.
    Float out_deriv; // Derivative of output function.
    double const dt;
    int const n_upstream;
    Vec delta, delta_T;
    Float bias_deriv;
    Float coup_scalar;
    Float bias;

    SAN& operator= (SAN const& rhs)
    {
        V = rhs.V;
        Nk = rhs.Nk;
        Ca = rhs.Ca;
        std::copy(rhs.delta.begin(), rhs.delta.end(), delta.begin());
        std::copy(rhs.delta_T.begin(), rhs.delta_T.end(), delta_T.begin());
        bias_deriv = rhs.bias_deriv;
        coup_scalar = rhs.coup_scalar;
        // NOTE: bias not set here, since we don't want to overwrite it when we reset the neurons.
        return *this;
    }

    SAN (int n_upstream_, double dt_, bool is_output)
        : V{(Float)0}, Nk{(Float)0}, Ca{(Float)0.1}, 
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
        // Calculate intermediate quantities
        Float const mCa_inf = 1 / (1 + exp(-(V + 20) / 9));
        Float const mKCa_inf = 1 / (1 + pow(kD / Ca, 3.5));
        Float const mNaP_inf = 1 / (1 + exp(-(V + 55.7) / 7.7));

        Float alpha_n = V == -34 ? 0.1 : 0.01 * (V + 34) / (1 - exp(-(V + 34) / 10));
        Float beta_n = 0.125 * exp(-(V + 44) / 25);

	// Alpha and beta are scaled by 4 in this model.
	alpha_n *= 4;
	beta_n *= 4;

        // Calculate leapfrog update for gating variable
	Nk = (alpha_n * dt + (1 - dt/2 * (alpha_n + beta_n)) * Nk) / (dt/2 * (alpha_n + beta_n) + 1);

        // Calculate leapfrog update for Ca2+ 
        Float const ICa = gCa * mCa_inf * mCa_inf * (V - vCa);
	Ca = (Ca * (1 - dt/2 * 1/tauCa) - dt/2 * alphaCa * 10 * A * ICa) / (1 + dt/2 * 1/tauCa);

        Float const G = gK*pow(Nk,4) + gCa*mCa_inf*mCa_inf + gKCa*mKCa_inf + gNaP*mNaP_inf*mNaP_inf*mNaP_inf + gL;
        Float const E = gK*pow(Nk,4)*vK + gCa*mCa_inf*mCa_inf*vCa + gKCa*mKCa_inf*vK + gNaP*mNaP_inf*mNaP_inf*mNaP_inf*vNa + gL*vL;

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
Float SAN::Ar = 5.0;
Float SAN::Ad = 0.18;
Float SAN::g_coupling = 0.01;
Float SAN::e_coupling_gaba = -70.0;

Float SAN::Tmax = 1.0;
Float SAN::A = 0.02;
Float SAN::vL = -60.95;
Float SAN::vNa = 55;
Float SAN::vK = -100;
Float SAN::tauhA = 15;
Float SAN::vCa = 120;
Float SAN::kD = 30;
Float SAN::vAMPA = 0;
Float SAN::vNMDA = 0;
Float SAN::vGABA = -70;
Float SAN::alphaCa = 0.5;
Float SAN::tauAMPA = 2;
Float SAN::tausNMDA = 100;
Float SAN::tauxNMDA = 2;
Float SAN::tauGABA = 10;
Float SAN::gL = 0.016307;
Float SAN::gNa = 12.2438;
Float SAN::gK = 19.20436;
Float SAN::gA = 1.79259;
Float SAN::gKS = 0.0350135;
Float SAN::gNaP = 0.63314;
Float SAN::gAR = 0.0166454;
Float SAN::gCa = 0.1624;
Float SAN::gKCa = 0.7506;
Float SAN::gAMPA = 0.513425;
Float SAN::gNMDA = 0.00434132;
Float SAN::gGABA = 0.00252916;
Float SAN::tauCa = 739.09;
Float SAN::a_hidden = 0;
Float SAN::a_out = 0;
Float SAN::LF_hidden = 0;
Float SAN::LF_out = 0;
Float SAN::use_bias = 0.0;
#endif
