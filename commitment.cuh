#ifndef COMMITMENT_CUH
#define COMMITMENT_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"

class Commitment: public G1TensorJacobian
{   
    public:
    using G1TensorJacobian::G1TensorJacobian;

    using G1TensorJacobian::operator+;
    using G1TensorJacobian::operator-;
    using G1TensorJacobian::operator*;
    using G1TensorJacobian::operator*=;

    G1TensorJacobian commit(const FrTensor& t) const;
    G1TensorJacobian commit_int (const FrTensor& t) const;
    G1TensorJacobian commit_int_multi(const vector<FrTensor>& t) const;

    Fr_t open(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u) const;

    // Variant of open() that captures the me_open proof transcript so stages
    // can write it into their proof bundle and an external verifier can check
    // the opening.
    Fr_t open_with_proof(const FrTensor& t, const G1TensorJacobian& c,
                         const vector<Fr_t>& u, vector<G1Jacobian_t>& proof_out) const;

    // Pedersen opening verifier. Returns true iff `proof` is a valid opening
    // of `com` at `u` to the claim `claim`. Folds `com` at the outer
    // challenges, runs the me_open round relation for log_2(size) steps, and
    // checks C_final == claim * g_final. Runs on GPU (single-thread kernel).
    bool verify_open(const G1TensorJacobian& com, const vector<Fr_t>& u,
                     Fr_t claim, const vector<G1Jacobian_t>& proof) const;

    static Commitment random(uint size);
    static Fr_t me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof);
};

struct Weight {
    Commitment generator;
    FrTensor weight;
    G1TensorJacobian com;
    uint in_dim;
    uint out_dim;
};

Weight create_weight(string generator_filename, string weight_filename, string com_filename, uint in_dim, uint out_dim);
// KERNEL void sum_axis_n_optimized(GLOBAL G1Jacobian_t* arr, GLOBAL G1Jacobian_t* arr_out, uint n, uint m);


KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators, Fr_t u, // always assume that scalars and u is in mont form
    GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
    GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0, GLOBAL G1Jacobian_t* temp_out1, 
    uint old_size, uint new_size);

#endif