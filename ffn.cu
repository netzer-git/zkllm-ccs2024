#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include "transcript.cuh"
#include "stage_proof.cuh"
#include <string>

// Absorb a weight's commitment tensor into the process-global Fiat-Shamir
// transcript. Must be called on every loaded Weight before any challenge
// is drawn.
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

    string input_file_name = argv[1];
    int seq_len = std::stoi(argv[2]);
    int embed_dim = std::stoi(argv[3]);
    int hidden_dim = std::stoi(argv[4]);
    string workdir = argv[5];
    string layer_prefix = argv[6];
    string output_file_name = argv[7];

    fs_transcript_init("ffn/" + layer_prefix);
    fs_absorb_bytes("input_fn", input_file_name.data(), input_file_name.size());
    fs_absorb_bytes("seq_len",   &seq_len,   sizeof(seq_len));
    fs_absorb_bytes("embed_dim", &embed_dim, sizeof(embed_dim));
    fs_absorb_bytes("hidden_dim",&hidden_dim,sizeof(hidden_dim));

    auto input_com = zkllm_stage::absorb_input_commitment_if_exists(input_file_name);

    auto up_proj = create_weight(
        workdir + "/mlp.up_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-commitment.bin",
        embed_dim,
        hidden_dim
    );

    auto gate_proj = create_weight(
        workdir + "/mlp.gate_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-commitment.bin",
        embed_dim,
        hidden_dim
    );

    auto down_proj = create_weight(
        workdir + "/mlp.down_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-commitment.bin",
        hidden_dim,
        embed_dim
    );

    // Bind the loaded weight commitments to the transcript before any
    // challenge is drawn.
    absorb_weight_commitment("ffn/up_proj.com",   up_proj);
    absorb_weight_commitment("ffn/gate_proj.com", gate_proj);
    absorb_weight_commitment("ffn/down_proj.com", down_proj);

    zkFC up_layer(embed_dim, hidden_dim, up_proj.weight);
    zkFC gate_layer(embed_dim, hidden_dim, gate_proj.weight);
    zkFC down_layer(hidden_dim, embed_dim, down_proj.weight);

    Rescaling up_rescale(1 << 16);
    Rescaling gate_rescale(1 << 20);
    Rescaling hidden_rescale(1 << 16);
    Rescaling down_rescale(1 << 16);

    FrTensor swiglu_values = FrTensor::from_int_bin("swiglu-table.bin");
    tLookupRangeMapping swiglu(-(1 << 21), 1 << 22, swiglu_values);

    // Commit the SwiGLU lookup table before any segment challenge is drawn
    swiglu.commit_table(gate_proj.generator);

    FrTensor input = FrTensor::from_int_bin(input_file_name);
    auto up_out = up_layer(input);
    auto up_out_ = up_rescale(up_out);


    auto gate_out = gate_layer(input);
    auto gate_out_ = gate_rescale(gate_out);
    auto p = swiglu(gate_out_);

    auto &swiglu_out = p.first, &swiglu_m = p.second;

    // Bind every SwiGLU witness tensor into the transcript before any
    // SwiGLU-related challenge is drawn.
    absorb_fr_tensor("ffn/up_out",     up_out_);
    absorb_fr_tensor("ffn/gate_out",   gate_out_);
    absorb_fr_tensor("ffn/swiglu_out", swiglu_out);
    absorb_fr_tensor("ffn/swiglu_m",   swiglu_m);

    auto temp_rand = fs_challenge_vec("ffn/swiglu_seg", 3);
    auto swiglu_u = fs_challenge_vec("ffn/swiglu_u", ceilLog2(seq_len * hidden_dim));
    auto swiglu_v = fs_challenge_vec("ffn/swiglu_v", ceilLog2(seq_len * hidden_dim));
    vector<Polynomial> swiglu_proof;
    

    auto down_in = swiglu_out * up_out_;
    auto down_in_ = hidden_rescale(down_in);

    // Prove the SwiGLU Hadamard product: down_in = swiglu_out (.) up_out_.
    {
        uint log_n = ceilLog2(down_in.size);
        auto hp_u = fs_challenge_vec("ffn/hadamard_u", log_n);
        auto hp_v = fs_challenge_vec("ffn/hadamard_v", log_n);
        auto hp_proof = hadamard_product_sumcheck(swiglu_out, up_out_, hp_u, hp_v);
        cout << "FFN Hadamard proof rounds: " << hp_proof.size() << endl;
    }


    auto down_out = down_layer(down_in_);
    auto down_out_ = down_rescale(down_out);

    down_out_.save_int(output_file_name);

    down_rescale.prove(down_out, down_out_, down_proj.generator);
    verifyWeightClaim(down_proj, down_layer.prove(down_in_, down_out)[0]);

    hidden_rescale.prove(down_in, down_in_, down_proj.generator);
    swiglu.prove(gate_out_, swiglu_out, swiglu_m, temp_rand[0], temp_rand[1], temp_rand[2], swiglu_u, swiglu_v, swiglu_proof, gate_proj.generator);
    cout << "SwiGLU proof complete." << endl;
    gate_rescale.prove(gate_out, gate_out_, gate_proj.generator);
    verifyWeightClaim(gate_proj, gate_layer.prove(input, gate_out)[0]);

    up_rescale.prove(up_out, up_out_, up_proj.generator);
    verifyWeightClaim(up_proj, up_layer.prove(input, up_out)[0]);

    auto output_com = zkllm_stage::commit_output_preferring_env(
        down_proj.generator, down_out_, output_file_name);
    zkllm_stage::write_stage_proof(
        output_file_name,
        "ffn/" + layer_prefix,
        input_file_name,
        input_com,
        output_com);

    return 0;
}