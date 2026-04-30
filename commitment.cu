#include "commitment.cuh"
#include <cstdint>

Commitment Commitment::random(uint size)
{
    Commitment out(size, G1Jacobian_generator);
    out *= FrTensor::random(size);
    return out; 
}

// KERNEL void com_sum_row_kernel(const G1Jacobian_t* arr, G1Jacobian_t* arr_out, uint m, uint n) {
//     auto row = GET_GLOBAL_ID();
//     if (row < m) {
//         G1Jacobian_t rowSum = arr[row * n];
//         for (uint i = 1; i < n; ++ i) {
//             rowSum = blstrs__g1__G1Affine_add(rowSum, arr[row * n + i]);
//         }
//         arr_out[row] = rowSum;
//     }
    
// }

G1TensorJacobian Commitment::commit(const FrTensor& t) const
{
    if (t.size % size != 0) throw std::runtime_error("Commitment::commit - Incompatible dimensions");

    uint m = t.size / size;
    G1TensorJacobian temp = (*this) * t;
    return temp.rowwise_sum(m, size);
}

DEVICE G1Jacobian_t commit_int_dev_func(G1Jacobian_t a, Fr_t s) {
    const int x = scalar_to_int(s);
    G1Jacobian_t out = blstrs__g1__G1Affine_ZERO;
    #pragma unroll
    for (uint i = 0; i < 31; ++ i) {
        if ((x >> i) & 1) out = blstrs__g1__G1Affine_add(out, a);
        a = blstrs__g1__G1Affine_double(a);
    }
    
    if (x < 0) out = blstrs__g1__G1Affine_add(out, G1Jacobian_minus(a));
    return out;
}

KERNEL void commit_int_kernel(const G1Jacobian_t* generators, const Fr_t* scalars, G1Jacobian_t* out, uint n, uint m) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= m * n) return;
    out[gid] = commit_int_dev_func(generators[gid % n], scalars[gid]);
}

G1TensorJacobian Commitment::commit_int (const FrTensor& t) const{
    if (t.size % size != 0) throw std::runtime_error("Commitment::commit_int - Incompatible dimensions");

    uint m = t.size / size;
    G1TensorJacobian temp(t.size);
    commit_int_kernel<<<(m*size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, temp.gpu_data, size, m);
    cudaDeviceSynchronize();
    return temp.rowwise_sum(m, size);
}

G1TensorJacobian Commitment::commit_int_multi(const vector<FrTensor>& ts) const{
    uint num_row = 0;
    for (auto& t : ts) {
        if (t.size % size != 0) throw std::runtime_error("Commitment::commit_int_multi - Incompatible dimensions");
        num_row += t.size / size;
    }

    G1TensorJacobian temp(num_row * size);
    auto temp_start = temp.gpu_data;
    for (auto& t: ts)
    {
        uint m = t.size / size;
        commit_int_kernel<<<(m*size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, temp_start, size, m);
        cudaDeviceSynchronize();
        temp_start += m * size;
    }
    return temp.rowwise_sum(temp.size / size, size);
}

KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators, Fr_t u, // always assume that scalars and u is in mont form
    GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
    GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0, GLOBAL G1Jacobian_t* temp_out1, 
    uint old_size, uint new_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;

    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    if (gid1 >= old_size) {
        new_scalars[gid] = blstrs__scalar__Scalar_sub(scalars[gid0], 
            blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, scalars[gid0]))
        );
        new_generators[gid] = G1Jacobian_mul(generators[gid0], u);
        temp_out[gid] = G1Jacobian_mul(generators[gid0], scalars[gid0]);
        temp_out0[gid] = blstrs__g1__G1Affine_ZERO;
        temp_out1[gid] = blstrs__g1__G1Affine_ZERO;
        return;
    }


    new_scalars[gid] = blstrs__scalar__Scalar_add(scalars[gid0], blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, blstrs__scalar__Scalar_sub(scalars[gid1], scalars[gid0]))));
    new_generators[gid] = blstrs__g1__G1Affine_add(generators[gid1], G1Jacobian_mul(blstrs__g1__G1Affine_add(generators[gid0], G1Jacobian_minus(generators[gid1])), u));
    temp_out[gid] = blstrs__g1__G1Affine_add(G1Jacobian_mul(generators[gid0], scalars[gid0]), G1Jacobian_mul(generators[gid1], scalars[gid1]));
    temp_out0[gid] = G1Jacobian_mul(generators[gid1], scalars[gid0]);
    temp_out1[gid] = G1Jacobian_mul(generators[gid0], scalars[gid1]);
}

Fr_t Commitment::me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof)
{
    if (t.size != generators.size) throw std::runtime_error("Commitment::me_open - Incompatible dimensions "+ std::to_string(t.size) + " " + std::to_string(generators.size));
    if (begin >= end)
    {
        proof.push_back(generators(0));
        return t(0);
    }
    uint new_size = (t.size + 1) / 2;
    FrTensor new_scalars(new_size);
    Commitment new_generators(new_size);
    G1TensorJacobian temp(new_size), temp0(new_size), temp1(new_size);
    me_open_step<<<(new_size+G1NumThread-1)/G1NumThread,G1NumThread>>>(t.gpu_data, generators.gpu_data, *begin, 
    new_scalars.gpu_data, new_generators.gpu_data, temp.gpu_data, temp0.gpu_data, temp1.gpu_data, 
    t.size, new_size);
    cudaDeviceSynchronize();
    proof.push_back(temp.sum());
    proof.push_back(temp0.sum());
    proof.push_back(temp1.sum());
    return me_open(new_scalars, new_generators, begin + 1, end, proof);
}



Fr_t Commitment::open(const FrTensor& t, const G1TensorJacobian& com, const vector<Fr_t>& u) const
{
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));
    auto g_temp = (com.size == 1)? com(0) : com(u_out);
    // if (size != (1 << u_in.size())) throw std::runtime_error("Incompatible dimensions");
    vector<G1Jacobian_t> proof;
    return me_open(t.partial_me(u_out, t.size / com.size), *this, u_in.begin(), u_in.end(), proof);
}

// Captures the me_open proof transcript instead of discarding it.
// Semantics otherwise identical to open().
Fr_t Commitment::open_with_proof(const FrTensor& t, const G1TensorJacobian& com,
                                 const vector<Fr_t>& u, vector<G1Jacobian_t>& proof_out) const
{
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));
    auto g_temp = (com.size == 1)? com(0) : com(u_out);
    return me_open(t.partial_me(u_out, t.size / com.size), *this, u_in.begin(), u_in.end(), proof_out);
}

// -----------------------------------------------------------------------------
// Pedersen opening verifier.
//
// The me_open prover emits, per round k, three aggregated G1 points
//   P0_k = Σ g_{2i}·s_{2i} + g_{2i+1}·s_{2i+1}     (this equals the current C_k)
//   P1_k = Σ g_{2i+1}·s_{2i}
//   P2_k = Σ g_{2i}·s_{2i+1}
// Given these, the folded commitment after applying challenge u is
//   C_{k+1} = (1-u)^2 · P1_k + u·(1-u) · P0_k + u^2 · P2_k.
// This identity is obtained by expanding (s_0+u(s_1-s_0))(g_1+u(g_0-g_1))
// pairwise and summing. The verifier must:
//   (a) check C_k == P0_k at every round (otherwise the prover's P0 claim is
//       inconsistent with its own folding, which would cascade); and
//   (b) run the recurrence; and
//   (c) at the leaf (size==1), check C_final == claim · g_final, where
//       g_final = proof.back() = generators(0) emitted by me_open's base case.
//
// The whole chain runs on GPU in a single-thread kernel (rounds are
// sequential; log(gen.size) <= ~20 so it's fast enough). This means the
// verifier uses the SAME arithmetic primitives the prover did, bit-exact.
// -----------------------------------------------------------------------------

KERNEL void me_open_verify_kernel(
    G1Jacobian_t c_start,
    const G1Jacobian_t* proof_g1,
    const Fr_t* u_in,
    Fr_t claim,
    uint L,
    uint8_t* ok_out)
{
    // Single-thread execution: rounds are sequential.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const Fr_t ONE_raw = {1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    G1Jacobian_t c = c_start;

    for (uint k = 0; k < L; ++k) {
        G1Jacobian_t P0 = proof_g1[3u * k + 0u];
        G1Jacobian_t P1 = proof_g1[3u * k + 1u];
        G1Jacobian_t P2 = proof_g1[3u * k + 2u];

        // (a) P0_k must equal the current folded commitment C_k.
        //     Point at infinity has z == Fp_ZERO (z coord all-zero limbs).
        G1Jacobian_t diff = blstrs__g1__G1Affine_add(c, G1Jacobian_minus(P0));
        bool z_is_zero = true;
        #pragma unroll
        for (uint i = 0; i < blstrs__fp__Fp_LIMBS; ++i) {
            if (diff.z.val[i] != 0u) { z_is_zero = false; }
        }
        if (!z_is_zero) { *ok_out = 0u; return; }

        // Build the three coefficients mod r. Challenge u is raw (non-Mont),
        // so we mirror me_open_step's Mont-form juggling: Scalar_mul yields
        // the Montgomery product, and Scalar_mont converts it back to raw.
        Fr_t u           = u_in[k];
        Fr_t one_minus_u = blstrs__scalar__Scalar_sub(ONE_raw, u);
        Fr_t omu_sq      = blstrs__scalar__Scalar_mont(
            blstrs__scalar__Scalar_mul(one_minus_u, one_minus_u));
        Fr_t u_sq        = blstrs__scalar__Scalar_mont(
            blstrs__scalar__Scalar_mul(u, u));
        Fr_t u_omu       = blstrs__scalar__Scalar_mont(
            blstrs__scalar__Scalar_mul(u, one_minus_u));

        // (b) C_{k+1} = omu^2 · P1 + u·omu · P0 + u^2 · P2.
        G1Jacobian_t t1 = G1Jacobian_mul(P1, omu_sq);
        G1Jacobian_t t2 = G1Jacobian_mul(P0, u_omu);
        G1Jacobian_t t3 = G1Jacobian_mul(P2, u_sq);
        c = blstrs__g1__G1Affine_add(
                blstrs__g1__G1Affine_add(t1, t2), t3);
    }

    // (c) Leaf: C_final ?= claim · g_final.
    G1Jacobian_t g_final  = proof_g1[3u * L];
    G1Jacobian_t expected = G1Jacobian_mul(g_final, claim);
    G1Jacobian_t dleaf    = blstrs__g1__G1Affine_add(c, G1Jacobian_minus(expected));
    bool z_is_zero = true;
    #pragma unroll
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; ++i) {
        if (dleaf.z.val[i] != 0u) { z_is_zero = false; }
    }
    *ok_out = z_is_zero ? 1u : 0u;
}

bool Commitment::verify_open(const G1TensorJacobian& com, const vector<Fr_t>& u,
                             Fr_t claim, const vector<G1Jacobian_t>& proof) const
{
    // ---- Shape checks --------------------------------------------------------
    if (com.size == 0 || size == 0) return false;
    // com.size and gen.size (= this->size) must be powers of 2 (or the
    // me_open path in the prover won't land on a length-1 leaf cleanly).
    auto is_pow2 = [](uint n) { return n != 0 && (n & (n - 1)) == 0; };
    if (!is_pow2(com.size) || !is_pow2(size)) return false;

    uint log_com = ceilLog2(com.size);      // |u_out|
    uint log_gen = ceilLog2(size);          // |u_in| = number of rounds L
    if (u.size() != log_com + log_gen) return false;
    if (proof.size() != 3ull * log_gen + 1ull) return false;

    // ---- Fold com at outer challenges to get C_start ------------------------
    vector<Fr_t> u_out(u.end() - log_com, u.end());
    vector<Fr_t> u_in (u.begin(), u.end() - log_com);
    G1Jacobian_t c_start = (com.size == 1) ? com(0) : com(u_out);

    // ---- Copy proof + challenges to device ----------------------------------
    G1Jacobian_t* d_proof = nullptr;
    Fr_t*         d_u     = nullptr;
    uint8_t*      d_ok    = nullptr;
    cudaMalloc(&d_proof, proof.size()  * sizeof(G1Jacobian_t));
    cudaMalloc(&d_u,     (u_in.empty() ? 1 : u_in.size()) * sizeof(Fr_t));
    cudaMalloc(&d_ok,    sizeof(uint8_t));

    cudaMemcpy(d_proof, proof.data(), proof.size() * sizeof(G1Jacobian_t),
               cudaMemcpyHostToDevice);
    if (!u_in.empty()) {
        cudaMemcpy(d_u, u_in.data(), u_in.size() * sizeof(Fr_t),
                   cudaMemcpyHostToDevice);
    }
    uint8_t zero = 0;
    cudaMemcpy(d_ok, &zero, sizeof(uint8_t), cudaMemcpyHostToDevice);

    // ---- Launch single-thread verifier --------------------------------------
    me_open_verify_kernel<<<1, 1>>>(c_start, d_proof, d_u, claim, log_gen, d_ok);
    cudaDeviceSynchronize();

    uint8_t ok = 0;
    cudaMemcpy(&ok, d_ok, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_proof);
    cudaFree(d_u);
    cudaFree(d_ok);
    return ok == 1u;
}

Weight create_weight(string generator_filename, string weight_filename, string com_filename, uint in_dim, uint out_dim) {
    Commitment generator(generator_filename);
    FrTensor weight = FrTensor::from_int_bin(weight_filename);
    G1TensorJacobian com(com_filename);
    return {generator, weight, com, in_dim, out_dim};
}