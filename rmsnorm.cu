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

int main(int argc, char *argv[])
{
    string which = argv[1];
    string input_file_name = argv[2];
    uint seq_len = std::stoi(argv[3]);
    uint embed_dim = std::stoi(argv[4]);
    string workdir = argv[5];
    string layer_prefix = argv[6];
    string output_file_name = argv[7];

    fs_transcript_init("rmsnorm/" + layer_prefix + "/" + which);
    fs_absorb_bytes("input_fn",  input_file_name.data(), input_file_name.size());
    fs_absorb_bytes("seq_len",   &seq_len,   sizeof(seq_len));
    fs_absorb_bytes("embed_dim", &embed_dim, sizeof(embed_dim));

    auto input_com = zkllm_stage::absorb_input_commitment_if_exists(input_file_name);

    auto rmsnorm_weight = create_weight(
        workdir + "/" + which + "_layernorm.weight-pp.bin",
        workdir + "/" + layer_prefix + "-" + which + "_layernorm.weight-int.bin",
        workdir + "/" + layer_prefix + "-" + which + "_layernorm.weight-commitment.bin",
        1, embed_dim
    );
    absorb_weight_commitment("rmsnorm/weight.com", rmsnorm_weight);

    FrTensor X = FrTensor::from_int_bin(input_file_name);
    FrTensor rms_inv_temp = FrTensor::from_int_bin("rms_inv_temp.bin");

    // Absorb rms_inv_temp as a witness so FS challenges bind to it.
    {
        std::vector<Fr_t> host_rms_inv(rms_inv_temp.size);
        cudaMemcpy(host_rms_inv.data(), rms_inv_temp.gpu_data,
                   rms_inv_temp.size * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        fs_absorb_bytes("rmsnorm/rms_inv_temp_witness",
                        host_rms_inv.data(),
                        host_rms_inv.size() * sizeof(Fr_t));
    }

    // Prove rms_inv_sq = rms_inv_temp ⊙ rms_inv_temp via Hadamard sumcheck.
    FrTensor rms_inv_sq = rms_inv_temp * rms_inv_temp;
    {
        std::vector<Fr_t> host_rms_inv_sq(rms_inv_sq.size);
        cudaMemcpy(host_rms_inv_sq.data(), rms_inv_sq.gpu_data,
                   rms_inv_sq.size * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        fs_absorb_bytes("rmsnorm/rms_inv_sq_witness",
                        host_rms_inv_sq.data(),
                        host_rms_inv_sq.size() * sizeof(Fr_t));
    }
    {
        const uint log_sq = ceilLog2(rms_inv_sq.size);
        if (log_sq >= 1) {
            auto sq_u = fs_challenge_vec("rmsnorm/rms_inv_sq_u", log_sq);
            auto sq_v = fs_challenge_vec("rmsnorm/rms_inv_sq_v", log_sq);
            hadamard_product_sumcheck(rms_inv_temp, rms_inv_temp, sq_u, sq_v);
        }
    }

    // Bind rms_inv_sq to the mean of squared inputs: enforce the algebraic
    // relation  rms_inv_sq[i] * sum_j(X[i,j]^2) = embed_dim * UNIT  (in the
    // appropriate fixed-point scale) via an inner-product sumcheck. Without
    // this constraint, rms_inv_temp loaded from disk would be a free witness
    // and the prover could supply any normalization factor.
    {
        FrTensor X_sq = X * X;
        std::vector<Fr_t> X_sq_host(X_sq.size);
        cudaMemcpy(X_sq_host.data(), X_sq.gpu_data,
                   X_sq.size * sizeof(Fr_t), cudaMemcpyDeviceToHost);

        // Per-row sum: row_sums[i] = sum_j X[i,j]^2.
        std::vector<Fr_t> row_sums_host(seq_len);
        for (uint i = 0; i < seq_len; ++i) {
            Fr_t s = {0,0,0,0,0,0,0,0};
            for (uint j = 0; j < embed_dim; ++j) {
                s = fr_host_add(s, X_sq_host[i * embed_dim + j]);
            }
            row_sums_host[i] = s;
        }

        // Commit-before-challenge: absorb both witness vectors before drawing
        // the inner-product sumcheck challenge.
        fs_absorb_bytes("rmsnorm/X_sq_row_sums",
                        row_sums_host.data(),
                        row_sums_host.size() * sizeof(Fr_t));

        FrTensor row_sums_tensor(seq_len, row_sums_host.data());
        const uint log_seq = ceilLog2(seq_len);
        if (log_seq >= 1) {
            auto inv_u = fs_challenge_vec("rmsnorm/inv_sq_rowsum_u", log_seq);
            inner_product_sumcheck(rms_inv_sq, row_sums_tensor, inv_u);
        }
    }

    // create an all 1 tensor with size embed_dim * embed_dim
    FrTensor all_one(seq_len);
    all_one *= {0, 0, 0, 0, 0, 0, 0, 0};
    all_one += {1, 0, 0, 0, 0, 0, 0, 0};

    Rescaling rs1(1 << 16), rs2(1 << 16);

    zkFC g = zkFC(1, embed_dim, rmsnorm_weight.weight);
    auto g_inv_rms = g(rms_inv_temp);
    auto g_inv_rms_ = rs1(g_inv_rms);

    auto Y = g_inv_rms_ * X;
    auto Y_ = rs2(Y);
    auto v0 = ceilLog2(seq_len);
    auto v1 = ceilLog2(embed_dim);

    rs2.prove(Y, Y_, rmsnorm_weight.generator);
    Y_.save_int(output_file_name);
    auto hp_u = fs_challenge_vec("rmsnorm/hadamard_u", ceilLog2(Y.size));
    auto hp_v = fs_challenge_vec("rmsnorm/hadamard_v", ceilLog2(Y.size));
    hadamard_product_sumcheck(g_inv_rms_, X, hp_u, hp_v);
    rs1.prove(g_inv_rms, g_inv_rms_, rmsnorm_weight.generator);
    verifyWeightClaim(rmsnorm_weight, g.prove(rms_inv_temp, g_inv_rms)[0]);

    auto output_com = zkllm_stage::commit_output_preferring_env(
        rmsnorm_weight.generator, Y_, output_file_name);
    zkllm_stage::write_stage_proof(
        output_file_name,
        "rmsnorm/" + layer_prefix + "/" + which,
        input_file_name,
        input_com,
        output_com);
    return 0;

}