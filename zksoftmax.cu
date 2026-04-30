#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "transcript.cuh"

DEVICE Fr_t zksoftmax_calculate_table_entry(uint x, double theta_k, double scaling_factor_in, double d, double Bk){
    unsigned long result = static_cast<unsigned long> (exp(log(theta_k) - ( Bk / (scaling_factor_in * sqrt(d))) * x) + 0.5);
    return {static_cast<uint> (result), static_cast<uint> (result >> 32), 0, 0, 0, 0, 0, 0};
}

KERNEL void zksoftmax_calculate_table(Fr_t* table, double theta_k, double scaling_factor_in, double d, double Bk, uint bk){
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < bk){
        table[x] = zksoftmax_calculate_table_entry(x, theta_k, scaling_factor_in, d, Bk);
    }
}


// 1 thread per row
// kernel to do first step of softmax, calculate hat{z} by row
// input: Fr_t *A, output: z hat as an array
KERNEL void zksoftmax_shift(const Fr_t *A, Fr_t *shift, Fr_t* out, uint rows, uint cols, double scaling_factor_in, double d){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows) {
        double temp = 0.0;
        for (int i = 0; i < cols; ++i) {
            temp += exp(static_cast<double> (scalar_to_long(A[idx * cols + i])) / (scaling_factor_in * sqrt(d)));
        }
        shift[idx] = double_to_scalar(scaling_factor_in * sqrt(d) * log(temp), 1UL);
        for (int i = 0; i < cols; ++i) {
            out[idx * cols + i] = blstrs__scalar__Scalar_sub(A[idx * cols + i], shift[idx]);
        }
    }
}

zkSoftmax::zkSoftmax(const vector<uint>& bs, uint L, uint M, unsigned long scaling_factor_in, const vector<double>& thetas, uint m, uint n, uint d, uint E):
bs(bs), K(bs.size()), L(L), M(M), scaling_factor_in(scaling_factor_in), thetas(thetas), m(m), n(n), d(d), E(E), 
least_significant_segments(), other_segments()
{
    if (K - L - M != thetas.size()) throw std::invalid_argument("bs and thetas must have the same size");
    if (K <= L + M) throw std::invalid_argument("K must be greater than L + M");

    vector<unsigned long> Bs(K);
    Bs[0] = 1L;
    for (uint i = 1; i < bs.size(); i++)
        Bs[i] = Bs[i-1] * static_cast<unsigned long> (bs[i-1]);

    for (uint i = 0; i < L; ++ i)
    {
        least_significant_segments.push_back({0, bs[i]});
    }

    for (uint i = L; i < K - M; ++ i)
    {
        FrTensor mapped_values(bs[i]); 
        uint threads_per_block = 256;
        uint blocks_per_grid = (bs[i] + threads_per_block - 1) / threads_per_block;
        zksoftmax_calculate_table<<<blocks_per_grid, threads_per_block>>>(mapped_values.gpu_data, thetas[i - L], scaling_factor_in, d, Bs[i], bs[i]);
        cudaDeviceSynchronize();
        other_segments.push_back({0, bs[i], mapped_values});

        // cout << "======== Debug info for segment " << i << endl;
        // // copy mapped_values.gpu_data to cpu
        // Fr_t* mapped_values_cpu = new Fr_t[bs[i]];
        // cudaMemcpy(mapped_values_cpu, mapped_values.gpu_data, bs[i] * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        // for (uint j = 0; j < bs[i]; ++ j)
        // {
        //     cout << "table[" << j << "] = " << mapped_values_cpu[j] << endl;
        //     cout << "comparison:" << thetas[i - L] / exp( (Bs[i] * j) / (scaling_factor_in * sqrt(d))) << endl;
        // }
        // delete[] mapped_values_cpu;
    }

    for (uint i = K - M; i < K; ++ i)
    {
        // Top-segment indicator: every entry must be 1 (saturated)
        int* mvals = new int[bs[i]];
        std::fill_n(mvals, bs[i], 1);
        FrTensor mapped_values(bs[i], mvals);
        delete[] mvals;
        other_segments.push_back({0, bs[i], mapped_values});
        // cout << "======== Debug info for segment " << i << endl;
        // // copy mapped_values.gpu_data to cpu
        // Fr_t* mapped_values_cpu = new Fr_t[bs[i]];
        // cudaMemcpy(mapped_values_cpu, mapped_values.gpu_data, bs[i] * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        // for (uint j = 0; j < bs[i]; ++ j)
        // {
        //     cout << "table[" << j << "] = " << mapped_values_cpu[j] << endl;
        // }
        // delete[] mapped_values_cpu;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error (std::string("CUDA error at zkSoftmax::zkSoftmax:") + cudaGetErrorString(err));
    }

}

void zkSoftmax::commit_tables(const Commitment& gen)
{
    for (auto& seg : least_significant_segments) {
        seg.commit_table(gen);
    }
    for (auto& seg : other_segments) {
        seg.commit_table(gen);
    }
}

KERNEL void zksoftmax_decompose_kernel(const Fr_t* X, const uint* bs, Fr_t* X_decomposed, uint N, uint K)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= N) return;
    auto minus_X = blstrs__scalar__Scalar_sub({0, 0, 0, 0, 0, 0, 0, 0}, X[gid]);
    if (!minus_X.val[7] && !minus_X.val[6] && !minus_X.val[5] && !minus_X.val[4] && !minus_X.val[3] && !minus_X.val[2])
    {
        unsigned long tmp = (static_cast<unsigned long> (minus_X.val[1]) << 32) | static_cast<unsigned long> (minus_X.val[0]);
        for (uint i = 0; i < K; ++ i)
        {
            X_decomposed[i * N + gid] = {static_cast <uint>(tmp % bs[i]), 0, 0, 0, 0, 0, 0, 0};
            tmp /= bs[i];
        }   
    }
}

// input should be m * n
FrTensor zkSoftmax::compute(const FrTensor& X, FrTensor& shift, FrTensor& X_shifted, vector<FrTensor>& X_segments, vector<FrTensor>& Y_segments, vector<FrTensor>& m_segments)
{
    if (X.size != m * n) throw std::invalid_argument("X must be m * n");
    if (shift.size != m) throw std::invalid_argument("shift must be m");
    if (X_shifted.size != X.size) throw std::invalid_argument("X_shifted must be the same size as X");

    // calculate shift
    uint threads_per_block = 256;
    uint blocks_per_grid = (m + threads_per_block - 1) / threads_per_block;
    zksoftmax_shift<<<blocks_per_grid, threads_per_block>>>(X.gpu_data, shift.gpu_data, X_shifted.gpu_data, m, n, scaling_factor_in, d);
    cudaDeviceSynchronize();

    // decompose X into segments
    Fr_t* X_decomposed;
    cudaMalloc(&X_decomposed, m * n * sizeof(Fr_t) * K);

    Fr_t** X_decomposed_segments = new Fr_t*[K];
    for (uint i = 0; i < K; ++ i) X_decomposed_segments[i] = X_decomposed + i * m * n;

    uint* bs_gpu;
    cudaMalloc(&bs_gpu, K * sizeof(uint));
    cudaMemcpy(bs_gpu, bs.data(), K * sizeof(uint), cudaMemcpyHostToDevice);

    blocks_per_grid = (m * n + threads_per_block - 1) / threads_per_block;
    zksoftmax_decompose_kernel<<<blocks_per_grid, threads_per_block>>>(X_shifted.gpu_data, bs_gpu, X_decomposed, m * n, K);
    cudaDeviceSynchronize();

    FrTensor out(m * n);
    cudaMemset(out.gpu_data, 0, out.size * sizeof(Fr_t));
    out += {1, 0, 0, 0, 0, 0, 0, 0};

    for (uint i = 0; i < L; ++ i)
    {   
        X_segments.push_back({m * n});
        cudaMemcpy(X_segments[i].gpu_data, X_decomposed_segments[i], out.size * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
        auto counts = least_significant_segments[i].prep(X_segments[i]);
        m_segments.push_back(counts);
    }

    for (uint i = L; i < K; ++ i)
    {
        X_segments.push_back({m * n});
        cudaMemcpy(X_segments[i].gpu_data, X_decomposed_segments[i], out.size * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
        auto p = other_segments[i - L](X_segments[i]);
        Y_segments.push_back(p.first);
        m_segments.push_back(p.second);
        out *= p.first;
    }

    delete[] X_decomposed_segments;
    cudaFree(X_decomposed);
    cudaFree(bs_gpu);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error (std::string("CUDA error at zkSoftmax::compute:") + cudaGetErrorString(err));
    }

    // cout << "out.size = " << out.size << endl;

    return out;
}

Fr_t zkSoftmax::prove(const FrTensor& Y, const FrTensor& X, const FrTensor& shift, const FrTensor& X_shifted,
    const vector<FrTensor>& X_segments, const vector<FrTensor>& Y_segments, const vector<FrTensor>& m_segments,
    const vector<Fr_t>& u_Y, const vector<Fr_t>& v_Y,
    const Fr_t& r_seg, const Fr_t& alpha_seg, const Fr_t& beta_seg,
    vector<Polynomial>& proof,
    const Commitment& gen)
{
    uint N = Y.size;
    if (X.size != N) throw std::invalid_argument("X and Y must have the same size "+to_string(X.size)+" "+to_string(Y.size));

    // -------------------------------------------------------------------------
    // Softmax row-sum / normalization check.
    //
    // The paper requires each row of Y to satisfy
    //     sum_j Y[i, j]  in  [theta - E, theta + E]
    // where `theta` is the output scaling factor for the dominant segment.
    // -------------------------------------------------------------------------
    if (N == m * n) {
        // Pull Y to host once.
        std::vector<Fr_t> Y_host(N);
        cudaMemcpy(Y_host.data(), Y.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        std::vector<Fr_t> row_sums(m);
        for (uint i = 0; i < m; ++i) {
            Fr_t s = {0,0,0,0,0,0,0,0};
            for (uint j = 0; j < n; ++j) {
                s = fr_host_add(s, Y_host[i * n + j]);
            }
            row_sums[i] = s;
        }
        // Bind row sums into the transcript so every challenge below depends
        // on them (commit-before-challenge discipline).
        fs_absorb_fr_vec("zksoftmax/row_sums", row_sums);
        // Local tolerance check: each row-sum must lie in [theta - E, theta + E].
        // Interpret row_sums[i] - theta as a signed long via the standard
        // "top-half-of-P is negative" convention and bound it by E.
        if (!thetas.empty() && E > 0) {
            // theta is a small public integer; build it directly as an Fr_t.
            // thetas.back() comes from the per-segment scaling, which for the
            // LLaMA-softmax build path is a small integer (<= 2^30 in normal
            // use). We conservatively accept it up to 2^62.
            long theta_l = static_cast<long>(thetas.back());
            if (theta_l < 0) theta_l = 0;
            // Build theta as a field element via repeated add (cheap since
            // it's only a few dozen bits).
            Fr_t theta_fr = {0,0,0,0,0,0,0,0};
            theta_fr.val[0] = static_cast<uint32_t>(theta_l & 0xFFFFFFFFu);
            theta_fr.val[1] = static_cast<uint32_t>((static_cast<uint64_t>(theta_l) >> 32) & 0xFFFFFFFFu);
            for (uint i = 0; i < m; ++i) {
                Fr_t diff = fr_host_sub(row_sums[i], theta_fr);
                long diff_l = 0;
                if (!fr_host_to_long(diff, diff_l)) {
                    throw std::runtime_error(
                        "zkSoftmax::prove: row-sum deviation out of long range at row "
                        + std::to_string(i) +
                        + ". Y does not normalize.");
                }
                if (diff_l < -static_cast<long>(E) || diff_l > static_cast<long>(E)) {
                    throw std::runtime_error(
                        "zkSoftmax::prove: row-sum tolerance check failed at row "
                        + std::to_string(i) + " (deviation=" + std::to_string(diff_l)
                        + ", E=" + std::to_string(E) + ").");
                }
            }
        }
    }

    // Y and Y_segments
    auto claim_Y = Y(u_Y);
    auto claim_Y_segs = multi_hadamard_sumchecks(claim_Y, Y_segments, u_Y, v_Y, proof);
    vector<Fr_t> claim_lus;
    

    for (uint i = 0; i < L; ++ i)
    {
        // cout << "======== Proving segment "<< i << "========" << endl;
        claim_lus.push_back(least_significant_segments[i].prove(X_segments[i], m_segments[i], alpha_seg, beta_seg, u_Y, v_Y, proof, gen));
    }

    for (uint i = L; i < K; ++ i)
    {
        // cout << "======== Proving segment "<< i << "========" << endl;
        claim_lus.push_back(other_segments[i-L].prove(X_segments[i], Y_segments[i - L], m_segments[i], 
            r_seg, alpha_seg, beta_seg, u_Y, v_Y, proof, gen));
    }

    vector<Fr_t> Y_seg_claims, X_seg_claims;
    for (uint i = L; i < K; ++ i) Y_seg_claims.push_back(Y_segments[i - L](v_Y));
    for (uint i = 0; i < K; ++ i) X_seg_claims.push_back(X_segments[i](v_Y));

    Fr_t minus_X_shifted_claim = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = bs.size() - 1; i >= 0; -- i ){
        minus_X_shifted_claim = minus_X_shifted_claim * Fr_t({bs[i], 0, 0, 0, 0, 0, 0, 0}) + X_seg_claims[i];
    }

    // auto X_shifted_opening = X_shifted(v_Y);
    vector<Fr_t> v_Y_(v_Y.begin() + ceilLog2(n), v_Y.end()) ;
    auto shift_const_opening = shift(v_Y_);

    auto X_opening = X(v_Y);

    if (shift_const_opening - minus_X_shifted_claim != X_opening) throw std::runtime_error("X claim is not correct");
    return X_opening;
}

zkAttn::zkAttn(unsigned long sf_Q, unsigned long sf_K, const vector<uint>& bs, uint L, uint M, const vector<double>& thetas, uint m, uint n, uint d, uint E):
zkSoftmax(bs, L, M, sf_Q * sf_K, thetas, m, n, d, E) {}

FrTensor zkAttn::compute(const FrTensor& Q, const FrTensor& K, const FrTensor& V, FrTensor& sm_in, FrTensor& sm_out,
    FrTensor& sm_shift, FrTensor& sm_in_shifted, vector<FrTensor>& sm_in_segments, vector<FrTensor>& sm_out_segments, vector<FrTensor>& sm_m_segments)
{
    if (Q.size != m * d) throw std::invalid_argument("Q must be m * d");
    if (K.size != n * d) throw std::invalid_argument("K must be n * d");
    if (V.size != n * d) throw std::invalid_argument("V must be n * d");
    
    sm_in = FrTensor::matmul(Q, K.transpose(n, d), m, d, n);
    // cout << sm_in(0) << " " << sm_in(sm_in.size -1) << endl;
    sm_out = zkSoftmax::compute(sm_in, sm_shift, sm_in_shifted, sm_in_segments, sm_out_segments, sm_m_segments);
    // cout << sm_out(0) << " " << sm_out(sm_out.size -1) << endl;
    return FrTensor::matmul(sm_out, V, m, n, d);
}

vector<Claim> zkAttn::prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const vector<FrTensor>& sm_in_segments, const vector<FrTensor>& sm_out_segments, const vector<FrTensor>& sm_m_segments,
        const Commitment& gen)
{
    // Challenges are derived from the Fiat-Shamir transcript (initialized by
    // the enclosing stage's main()). The caller is responsible for absorbing
    // Q, K, V, and out commitments into the transcript before invoking
    // prove().
    auto u_matmul_out = fs_challenge_vec("zkattn/u_matmul_out", ceilLog2(m));
    auto v_matmul_out = fs_challenge_vec("zkattn/v_matmul_out", ceilLog2(n));
    auto w_matmul_out = fs_challenge_vec("zkattn/w_matmul_out", ceilLog2(d));

    auto v_sm = fs_challenge_vec("zkattn/v_sm", ceilLog2(m) + ceilLog2(n));
    auto temp_rand = fs_challenge_vec("zkattn/seg_challenges", 3);
    auto &r_seg = temp_rand[0], &alpha_seg = temp_rand[1], &beta_seg = temp_rand[2];
    auto v_matmul_in = fs_challenge_vec("zkattn/v_matmul_in", ceilLog2(d));
    vector<Polynomial> proof;
    this -> prove(Q, K, V, out, sm_out, sm_in, sm_shift, sm_in_shifted, sm_in_segments, sm_out_segments, sm_m_segments, 
        u_matmul_out, v_matmul_out, w_matmul_out, v_sm, r_seg, alpha_seg, beta_seg, v_matmul_in, proof, gen);
    cout << "zkAttn proof complete." << endl;
    return {};
}

Fr_t zkAttn::prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const vector<FrTensor>& sm_in_segments, const vector<FrTensor>& sm_out_segments, const vector<FrTensor>& sm_m_segments,
        const vector<Fr_t>& u_matmul_out, const vector<Fr_t>& v_matmul_out, const vector<Fr_t>& w_matmul_out, 
        const vector<Fr_t>& v_sm, const Fr_t& r_seg, const Fr_t& alpha_seg, const Fr_t& beta_seg, 
        const vector<Fr_t>& v_matmul_in,
        vector<Polynomial>& proof,
        const Commitment& gen)
{
    auto out_claim = out.multi_dim_me({u_matmul_out, w_matmul_out}, {m, d});
    auto out_vec0 = sm_out.partial_me(u_matmul_out, m, n);
    auto out_vec1 = V.partial_me(w_matmul_out, d, 1);
    auto out_matmul_claim = zkip(out_claim, out_vec0, out_vec1, v_matmul_out, proof);

    

    auto V_claim = V.multi_dim_me({v_matmul_out, w_matmul_out}, {n, d});
    auto sm_out_claim = sm_out.multi_dim_me({u_matmul_out, v_matmul_out}, {m, n});

    if (out_matmul_claim != V_claim * sm_out_claim) throw std::runtime_error("out_matmul_claim is not correct");

    auto sm_in_claim = zkSoftmax::prove(sm_out, sm_in, sm_shift, sm_in_shifted, sm_in_segments, sm_out_segments, sm_m_segments,
        concatenate<Fr_t>({v_matmul_out, u_matmul_out}), v_sm, r_seg, alpha_seg, beta_seg, proof, gen);

    vector<Fr_t> u_matmul_in (v_sm.end() - ceilLog2(m), v_sm.end());
    vector<Fr_t> w_matmul_in (v_sm.begin(), v_sm.end() - ceilLog2(m)); 
    auto q = Q.partial_me(u_matmul_in, m, d);
    auto k = K.partial_me(w_matmul_in, n, d); // TRANSPOSE!!! Es correcto!
    return zkip(sm_in_claim, q, k, v_matmul_in, proof);
}

zkAttnStacked::zkAttnStacked(uint num, unsigned long sf_Q, unsigned long sf_K, const vector<uint>& bs, uint L, uint M, const vector<double>& thetas, uint m, uint n, uint d, uint E): 
num(num), zkAttn(sf_Q, sf_K, bs, L, M, thetas, m, n, d, E) {}

Fr_t zkAttnStacked::prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const vector<FrTensor>& sm_in_segments, const vector<FrTensor>& sm_out_segments, const vector<FrTensor>& sm_m_segments,
        const vector<Fr_t>& u_matmul_out_num, const vector<Fr_t>& v_matmul_out_num,
        const vector<Fr_t>& u_matmul_out, const vector<Fr_t>& v_matmul_out, const vector<Fr_t>& w_matmul_out, 
        const vector<Fr_t>& v_sm, const Fr_t& r_seg, const Fr_t& alpha_seg, const Fr_t& beta_seg, 
        const vector<Fr_t>& v_matmul_in_num, const vector<Fr_t>& v_matmul_in,
        vector<Polynomial>& proof,
        const Commitment& gen)
{
    auto out_claim = out.multi_dim_me({u_matmul_out_num, u_matmul_out, w_matmul_out}, {num, m, d});
    
    auto sm_out_reduced = sm_out.partial_me(u_matmul_out, m, n);
    auto V_reduced = V.partial_me(w_matmul_out, d, 1);
    auto out_matmul_claim = zkip_stacked(out_claim, sm_out_reduced, V_reduced, u_matmul_out_num, v_matmul_out, v_matmul_out_num, num, n, proof);

    auto V_claim = V.multi_dim_me({v_matmul_out_num, v_matmul_out, w_matmul_out}, {num, n, d});
    auto sm_out_claim = sm_out.multi_dim_me({v_matmul_out_num, u_matmul_out, v_matmul_out}, {num, m, n});

    if (out_matmul_claim != V_claim * sm_out_claim) throw std::runtime_error("out_matmul_claim is not correct");

    auto sm_in_claim = zkSoftmax::prove(sm_out, sm_in, sm_shift, sm_in_shifted, sm_in_segments, sm_out_segments, sm_m_segments,
        concatenate<Fr_t>({v_matmul_out, u_matmul_out, v_matmul_out_num}), v_sm, r_seg, alpha_seg, beta_seg, proof, gen);

    vector<Fr_t> u_matmul_in_num (v_sm.end() - ceilLog2(num), v_sm.end());
    vector<Fr_t> u_matmul_in (v_sm.end() - ceilLog2(num) - ceilLog2(m), v_sm.end() - ceilLog2(num));
    vector<Fr_t> w_matmul_in (v_sm.begin(), v_sm.end() - ceilLog2(num) - ceilLog2(m)); 
    auto Q_reduced = Q.partial_me(u_matmul_in, m, d);
    auto K_reduced = K.partial_me(w_matmul_in, n, d);
    return zkip_stacked(sm_in_claim, Q_reduced, K_reduced, u_matmul_in_num, v_matmul_in, v_matmul_in_num, num, d, proof);
}