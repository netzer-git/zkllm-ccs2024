#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include "transcript.cuh"
#include "stage_proof.cuh"
#include <string>

// Absorb a weight's commitment tensor into the process-global FS transcript.
static void absorb_weight_commitment(const char* label, const Weight& w) {
    std::vector<G1Jacobian_t> host(w.com.size);
    cudaMemcpy(host.data(), w.com.gpu_data,
               w.com.size * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    fs_absorb_bytes(label, host.data(), host.size() * sizeof(G1Jacobian_t));
}

// Absorb the raw field-element contents of an FrTensor into the transcript.
static void absorb_fr_tensor(const char* label, const FrTensor& t) {
    if (t.size == 0) {
        fs_absorb_bytes(label, nullptr, 0);
        return;
    }
    std::vector<Fr_t> host(t.size);
    cudaMemcpy(host.data(), t.gpu_data,
               t.size * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    fs_absorb_bytes(label, host.data(), host.size() * sizeof(Fr_t));
}

int main(int argc, char *argv[])
{
    string mode = argv[1];
    string input_file_name = argv[2];
    uint seq_len = std::stoi(argv[3]);
    uint embed_dim = std::stoi(argv[4]);
    string workdir = argv[5];
    string layer_prefix = argv[6];
    string output_file_name = argv[7];

    fs_transcript_init("self-attn/" + layer_prefix + "/" + mode);
    fs_absorb_bytes("input_fn",  input_file_name.data(), input_file_name.size());
    fs_absorb_bytes("seq_len",   &seq_len,   sizeof(seq_len));
    fs_absorb_bytes("embed_dim", &embed_dim, sizeof(embed_dim));

    // Absorb prior stage's output commitment.
    auto input_com = zkllm_stage::absorb_input_commitment_if_exists(input_file_name);

    if (mode == "linear")
    {
        auto q_proj = create_weight(
            workdir + "/self_attn.q_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-commitment.bin",
            embed_dim,
            embed_dim
        );

        auto k_proj = create_weight(
            workdir + "/self_attn.k_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-commitment.bin",
            embed_dim,
            embed_dim
        );

        auto v_proj = create_weight(
            workdir + "/self_attn.v_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-commitment.bin",
            embed_dim,
            embed_dim
        );
        // Bind QKV weight commitments to transcript before any challenge
        // is drawn.
        absorb_weight_commitment("self_attn/q_proj.com", q_proj);
        absorb_weight_commitment("self_attn/k_proj.com", k_proj);
        absorb_weight_commitment("self_attn/v_proj.com", v_proj);

        zkFC q_layer(embed_dim, embed_dim, q_proj.weight);
        zkFC k_layer(embed_dim, embed_dim, k_proj.weight);
        zkFC v_layer(embed_dim, embed_dim, v_proj.weight);
        Rescaling q_rescale(1 << 16);
        Rescaling k_rescale(1 << 16);
        Rescaling v_rescale(1 << 16);

        FrTensor input = FrTensor::from_int_bin(input_file_name);
        auto Q = q_layer(input);
        auto Q_ = q_rescale(Q);

        auto K = k_layer(input);
        auto K_ = k_rescale(K);

        auto V = v_layer(input);
        auto V_ = v_rescale(V);
        
        q_rescale.prove(Q, Q_, q_proj.generator);
        k_rescale.prove(K, K_, k_proj.generator);
        v_rescale.prove(V, V_, v_proj.generator);

        verifyWeightClaim(k_proj, k_layer.prove(input, K)[0]);
        verifyWeightClaim(q_proj, q_layer.prove(input, Q)[0]);
        verifyWeightClaim(v_proj, v_layer.prove(input, V)[0]);

        Q_.save_int("temp_Q.bin");
        K_.save_int("temp_K.bin");
        V_.save_int("temp_V.bin");

        // Commit each temp file so the "attn" mode run can bind to them.
        auto q_com = zkllm_stage::commit_output_preferring_env(
            q_proj.generator, Q_, std::string("temp_Q.bin"));
        auto k_com = zkllm_stage::commit_output_preferring_env(
            k_proj.generator, K_, std::string("temp_K.bin"));
        auto v_com = zkllm_stage::commit_output_preferring_env(
            v_proj.generator, V_, std::string("temp_V.bin"));
        std::vector<G1Jacobian_t> combined_com;
        combined_com.insert(combined_com.end(), q_com.begin(), q_com.end());
        combined_com.insert(combined_com.end(), k_com.begin(), k_com.end());
        combined_com.insert(combined_com.end(), v_com.begin(), v_com.end());
        zkllm_stage::write_stage_proof(
            output_file_name,
            "self-attn/" + layer_prefix + "/linear",
            input_file_name,
            input_com,
            combined_com);

        cout << "QKV linear proof successfully verified!" << endl;

        return 0;
    }

    else if (mode == "attn")
    {
        // Absorb the three QKV .com handoffs from the prior "linear" run.
        auto qin_com = zkllm_stage::absorb_input_commitment_if_exists("temp_Q.bin");
        auto kin_com = zkllm_stage::absorb_input_commitment_if_exists("temp_K.bin");
        auto vin_com = zkllm_stage::absorb_input_commitment_if_exists("temp_V.bin");

        auto Q = FrTensor::from_int_bin("temp_Q.bin");
        auto K = FrTensor::from_int_bin("temp_K.bin");
        auto V = FrTensor::from_int_bin("temp_V.bin");
        auto d = Q.size / seq_len;
        
        auto X = FrTensor::matmul(Q, K.transpose(seq_len, d), seq_len, d, seq_len);

        zkSoftmax softmax({1<<8, 1<<20, 1<<20}, 1, 0, 1UL<<32, {1<<18, 1<<22}, seq_len, seq_len, d, 1);
        Rescaling rs1(1<< 20), rs2(1<<20);

        // Load the generator early — needed for rescaling proofs and the
        // output commitment at the end of this mode.
        Commitment attn_fallback_gen(
            workdir + "/self_attn.v_proj.weight-pp.bin");

        // Commit all softmax segment tables before any challenge is drawn
        softmax.commit_tables(attn_fallback_gen);

        FrTensor shift(seq_len), X_shifted(seq_len * seq_len);
        vector<FrTensor> X_segments, Y_segments, m_segments;
        FrTensor Y = softmax.compute(X, shift, X_shifted, X_segments, Y_segments, m_segments);    
        Y.save_long("temp_head_Y.bin");
        
        
        auto out = FrTensor::matmul(Y, V, seq_len, seq_len, d);
        auto out_ = rs2(out);
        auto out__ = rs1(out_);

        out__.save_int("temp_head_out.bin");

        rs1.prove(out_, out__, attn_fallback_gen);
        rs2.prove(out, out_, attn_fallback_gen);

        // Bind every softmax witness tensor into the transcript before any
        // softmax-related challenge is drawn, so the prover cannot adapt its
        // witness to the challenge.
        absorb_fr_tensor("self_attn/softmax_Y",         Y);
        absorb_fr_tensor("self_attn/softmax_shift",     shift);
        absorb_fr_tensor("self_attn/softmax_X_shifted", X_shifted);
        for (size_t i = 0; i < X_segments.size(); ++i) {
            std::string lbl = "self_attn/softmax_X_seg_" + std::to_string(i);
            absorb_fr_tensor(lbl.c_str(), X_segments[i]);
        }
        for (size_t i = 0; i < Y_segments.size(); ++i) {
            std::string lbl = "self_attn/softmax_Y_seg_" + std::to_string(i);
            absorb_fr_tensor(lbl.c_str(), Y_segments[i]);
        }
        for (size_t i = 0; i < m_segments.size(); ++i) {
            std::string lbl = "self_attn/softmax_m_seg_" + std::to_string(i);
            absorb_fr_tensor(lbl.c_str(), m_segments[i]);
        }

        auto temp_rand = fs_challenge_vec("self_attn/attn_seg", 3);
        vector<Polynomial> proof;
        auto u1 = fs_challenge_vec("self_attn/u1", ceilLog2(seq_len));
        auto u2 = fs_challenge_vec("self_attn/u2", ceilLog2(d));
        auto ud = fs_challenge_vec("self_attn/ud", ceilLog2(seq_len));
        auto claim = out.multi_dim_me({u1, u2}, {seq_len, d});
        auto final_claim = zkip(claim, Y.partial_me(u1, seq_len, seq_len), V.partial_me(u2, d, 1), ud, proof);

        auto sm_u  = fs_challenge_vec("self_attn/softmax_u",  ceilLog2(Y.size));
        auto sm_v  = fs_challenge_vec("self_attn/softmax_v",  ceilLog2(Y.size));
        softmax.prove(Y, X, shift, X_shifted, X_segments, Y_segments, m_segments,
            sm_u, sm_v, temp_rand[0], temp_rand[1], temp_rand[2], proof, attn_fallback_gen);
        auto u1_ = fs_challenge_vec("self_attn/u1_", ceilLog2(seq_len));
        auto u2_ = fs_challenge_vec("self_attn/u2_", ceilLog2(seq_len));
        auto ud_ = fs_challenge_vec("self_attn/ud_", ceilLog2(d));
        auto claim_ = X.multi_dim_me({u1_, u2_}, {seq_len, seq_len});
        auto final_claim_ = zkip(claim_, Q.partial_me(u1_, seq_len, d), K.partial_me(u2_, seq_len, d), ud_, proof);
        cout << "Self attention proof successfully verified!" << endl;

        // Commit the final attention output and record the stage bundle.
        auto output_com = zkllm_stage::commit_output_preferring_env(
            attn_fallback_gen, out__, std::string("temp_head_out.bin"));
        std::vector<G1Jacobian_t> combined_in;
        combined_in.insert(combined_in.end(), qin_com.begin(), qin_com.end());
        combined_in.insert(combined_in.end(), kin_com.begin(), kin_com.end());
        combined_in.insert(combined_in.end(), vin_com.begin(), vin_com.end());
        zkllm_stage::write_stage_proof(
            output_file_name,
            "self-attn/" + layer_prefix + "/attn",
            input_file_name,
            combined_in,
            output_com);
        return 0;
    }
    return 0;
}