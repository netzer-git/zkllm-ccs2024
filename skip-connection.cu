#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include "transcript.cuh"
#include "stage_proof.cuh"
#include <fstream>
#include <string>

// =============================================================================
// Skip connection stage: z = x + y.
//
// Protocol (all values mod r, the BLS12-381 scalar field):
//   1. Commit x, y, z with the public Pedersen generator.
//   2. Absorb Com(x), Com(y), Com(z) into the Fiat-Shamir transcript.
//   3. Derive challenge u <- FS(transcript) of length ceilLog2(x.size).
//   4. Open each commitment at u, capturing the me_open proof transcript.
//   5. Verify locally that z_hat == x_hat + y_hat. Because the multilinear
//      extension of (x + y) equals MLE(x) + MLE(y) pointwise, this single
//      check is a complete proof that z = x + y (no sumcheck needed; MLE
//      linearity replaces it).
//   6. Write the proof bundle to <output_file>.proof for the verifier.
//
// Arguments: block_input_fn block_output_fn output_fn generator_fn
// The generator file is required; if it is missing we fail hard rather than
// silently generate an unprovable output.
// =============================================================================

static void write_proof(const string& fn,
                        const G1TensorJacobian& com_x,
                        const G1TensorJacobian& com_y,
                        const G1TensorJacobian& com_z,
                        const vector<Fr_t>& u,
                        Fr_t x_hat, Fr_t y_hat, Fr_t z_hat,
                        const vector<G1Jacobian_t>& opx,
                        const vector<G1Jacobian_t>& opy,
                        const vector<G1Jacobian_t>& opz)
{
    std::ofstream f(fn, std::ios::binary);
    if (!f) throw std::runtime_error("skip-connection: cannot open proof file " + fn);

    auto wu32 = [&](uint32_t v){ f.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    auto wfr  = [&](const Fr_t& v){ f.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    auto wg1v = [&](const vector<G1Jacobian_t>& v){
        wu32(static_cast<uint32_t>(v.size()));
        if (!v.empty()) f.write(reinterpret_cast<const char*>(v.data()),
                                v.size() * sizeof(G1Jacobian_t));
    };

    // Magic + version.
    const char magic[8] = {'z','k','L','S','K','I','P','\0'};
    f.write(magic, 8);
    wu32(1); // version

    // Serialize commitments (sizes + raw Jacobian-coord bytes).
    auto wcom = [&](const G1TensorJacobian& c){
        wu32(c.size);
        vector<G1Jacobian_t> host(c.size);
        cudaMemcpy(host.data(), c.gpu_data, c.size * sizeof(G1Jacobian_t),
                   cudaMemcpyDeviceToHost);
        f.write(reinterpret_cast<const char*>(host.data()),
                host.size() * sizeof(G1Jacobian_t));
    };
    wcom(com_x); wcom(com_y); wcom(com_z);

    wu32(static_cast<uint32_t>(u.size()));
    for (const auto& ui : u) wfr(ui);

    wfr(x_hat); wfr(y_hat); wfr(z_hat);

    wg1v(opx); wg1v(opy); wg1v(opz);
}

int main(int argc, char *argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: skip-connection <block_input> <block_output> "
                     "<output> <generator>\n"
                     "  (generator file is REQUIRED in the fixed build to"
                     " produce a sound proof)" << std::endl;
        return 1;
    }
    string block_input_fn  = argv[1];
    string block_output_fn = argv[2];
    string output_fn       = argv[3];
    string generator_fn    = argv[4];

    // ---- Stage setup ---------------------------------------------------------
    fs_transcript_init("skip-connection");
    // Bind this stage's public inputs.
    fs_absorb_bytes("block_input_fn",  block_input_fn.data(),  block_input_fn.size());
    fs_absorb_bytes("block_output_fn", block_output_fn.data(), block_output_fn.size());

    // Absorb the two input commitments (from the upstream rmsnorm/ffn/self-attn
    // stages). If either .com is missing, a distinct marker is absorbed so an
    // external verifier can detect the gap.
    zkllm_stage::absorb_input_commitment_if_exists(block_input_fn);
    zkllm_stage::absorb_input_commitment_if_exists(block_output_fn);

    // ---- Load tensors + generator -------------------------------------------
    FrTensor x = FrTensor::from_int_bin(block_input_fn);
    FrTensor y = FrTensor::from_int_bin(block_output_fn);
    if (x.size != y.size)
        throw std::runtime_error("skip-connection: size mismatch x=" +
                                 std::to_string(x.size) + " y=" +
                                 std::to_string(y.size));
    FrTensor z = x + y;

    Commitment gen(generator_fn);
    if (x.size % gen.size != 0)
        throw std::runtime_error("skip-connection: tensor size " +
                                 std::to_string(x.size) +
                                 " not a multiple of generator size " +
                                 std::to_string(gen.size));

    // ---- Commit all three tensors -------------------------------------------
    // commit_int matches the convention in commit-param.cu for int-encoded
    // fixed-point tensors.
    auto com_x = gen.commit_int(x);
    auto com_y = gen.commit_int(y);
    auto com_z = gen.commit_int(z);

    // ---- Fiat-Shamir: absorb before drawing challenges -----------------------
    auto absorb_com = [](const char* label, const G1TensorJacobian& c) {
        vector<G1Jacobian_t> host(c.size);
        cudaMemcpy(host.data(), c.gpu_data, c.size * sizeof(G1Jacobian_t),
                   cudaMemcpyDeviceToHost);
        fs_absorb_bytes(label, host.data(), host.size() * sizeof(G1Jacobian_t));
    };
    absorb_com("com_x", com_x);
    absorb_com("com_y", com_y);
    absorb_com("com_z", com_z);

    // ---- Draw evaluation challenge ------------------------------------------
    uint log_n = ceilLog2(x.size);
    vector<Fr_t> u = fs_challenge_vec("skip_eval_point", log_n);

    // ---- Open all three at u, capturing opening proofs ----------------------
    vector<G1Jacobian_t> opx, opy, opz;
    Fr_t x_hat = gen.open_with_proof(x, com_x, u, opx);
    Fr_t y_hat = gen.open_with_proof(y, com_y, u, opy);
    Fr_t z_hat = gen.open_with_proof(z, com_z, u, opz);

    // ---- Local linearity check ----------------------------------------------
    // MLE(x+y)(u) = MLE(x)(u) + MLE(y)(u) holds identically. Any cheating
    // prover who sent com_z != commit(x+y) will, with probability 1 - O(1/|F|),
    // produce z_hat != x_hat + y_hat here.
    Fr_t sum_xy = fr_host_add(x_hat, y_hat);
    if (!(sum_xy == z_hat)) {
        std::cerr << "skip-connection: linearity check failed (z_hat != x_hat"
                     " + y_hat). Proof aborted." << std::endl;
        return 1;
    }

    // ---- Persist z and the proof bundle -------------------------------------
    z.save_int(output_fn);
    write_proof(output_fn + ".proof",
                com_x, com_y, com_z,
                u, x_hat, y_hat, z_hat,
                opx, opy, opz);

    // Also persist the output commitment as `<output>.com` so the downstream
    // stage can absorb it via the stage_proof handoff path (same convention
    // rmsnorm/ffn/self-attn follow).
    {
        std::vector<G1Jacobian_t> host(com_z.size);
        cudaMemcpy(host.data(), com_z.gpu_data,
                   com_z.size * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
        std::ofstream cf(output_fn + ".com", std::ios::binary);
        if (!cf) throw std::runtime_error(
            "skip-connection: cannot open " + output_fn + ".com");
        cf.write(reinterpret_cast<const char*>(host.data()),
                 host.size() * sizeof(G1Jacobian_t));
    }

    std::cout << "skip-connection proof OK (size=" << x.size
              << ", log_n=" << log_n << ")" << std::endl;
    return 0;
}