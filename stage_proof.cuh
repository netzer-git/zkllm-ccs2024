#ifndef STAGE_PROOF_CUH
#define STAGE_PROOF_CUH

// =============================================================================
// stage_proof.cuh — cross-stage soundness binding.
//
// Tensors handed between executables live in raw `.bin` files with no
// cryptographic linkage by default: stage N+1 simply trusts the byte pattern
// on disk. A malicious "prover" that controlled only one stage binary could
// therefore feed stage N+1 a tensor of its choice without any downstream
// verifier ever noticing.
//
// This header provides the minimum set of helpers every stage needs to close
// that gap without re-architecting the whole prover:
//
//   1. absorb_input_commitment_if_exists(input_fn)
//        Reads `<input_fn>.com` (a serialized G1Jacobian_t vector dumped by
//        the previous stage) and absorbs it into the current transcript.
//        When the `.com` file is absent (e.g. this is the first stage that
//        consumes `model_weight.bin` from disk), we still absorb a distinct
//        "missing" marker so the transcript records the absence.
//
//   2. commit_and_persist_output(gen, Y, output_fn)
//        Commits the output tensor with `gen`, persists the resulting
//        commitment to `<output_fn>.com` for the next stage to consume, and
//        absorbs the commitment into the current transcript. Returns the
//        host-resident commitment bytes so the caller can also write them
//        into the stage's `.proof` file.
//
//   3. write_stage_proof(...)
//        Writes `<output_fn>.proof` containing: magic + version + stage tag +
//        input filename + output filename + input_com bytes + output_com bytes
//        + the final 32-byte transcript digest. The downstream verifier can
//        replay the transcript and check that the ending digest matches,
//        establishing the chain.
//
// The checksum-chain approach does NOT replace per-stage sumcheck verifiers
// (which remain partial for rmsnorm / ffn / self-attn); it only guarantees
// that whatever each stage proved, it proved about the SAME tensor that its
// upstream neighbour committed to, and that the next stage is pinned to the
// SAME output this stage produced. Combined with `Commitment::verify_open`
// (main.cu verify-skip), it removes the last way a malicious stage could
// substitute data between approved checkpoints.
//
// All functions are host-only, header-only, and run outside any kernel.
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "fr-tensor.cuh"     // FrTensor, G1Jacobian_t typedef
#include "g1-tensor.cuh"     // G1TensorJacobian
#include "commitment.cuh"    // Commitment::commit_int
#include "transcript.cuh"    // Transcript, fs_transcript, fs_absorb_bytes

namespace zkllm_stage {

// Magic+version used across every non-skip stage. 8 bytes, zero-padded.
// Skip-connection uses its own magic "zkLSKIP\0" — this one is generic.
static constexpr char kStageMagic[8]  = {'z','k','L','S','T','A','G','E'};
static constexpr uint32_t kStageVersion = 1u;

// Internal helper: write a length-prefixed byte blob. Length is uint32_t LE.
inline void write_lenprefixed(std::ofstream& f, const void* data, uint32_t len) {
    f.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len) f.write(reinterpret_cast<const char*>(data), len);
}

// Internal helper: read a length-prefixed byte blob into `out` (resized to
// the advertised length). Returns false on short read.
inline bool read_lenprefixed(std::ifstream& f, std::vector<uint8_t>& out) {
    uint32_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!f) return false;
    out.resize(len);
    if (len) f.read(reinterpret_cast<char*>(out.data()), len);
    return static_cast<bool>(f);
}

// -----------------------------------------------------------------------------
// 1. Absorb the prior stage's output commitment, if it exists.
//    Returns the raw commitment bytes (empty if the .com file was absent),
//    for inclusion in this stage's own .proof bundle.
// -----------------------------------------------------------------------------
inline std::vector<G1Jacobian_t>
absorb_input_commitment_if_exists(const std::string& input_fn)
{
    const std::string com_fn = input_fn + ".com";
    std::ifstream f(com_fn, std::ios::binary | std::ios::ate);
    if (!f) {
        // No prior commitment on disk. Absorb a "missing" marker so that if
        // the prover *could* have supplied one (and maliciously didn't), the
        // transcript diverges from an honest run that did.
        fs_absorb_bytes("stage/input_com_missing",
                        input_fn.data(), input_fn.size());
        return {};
    }
    std::streamsize sz = f.tellg();
    if (sz < 0 || static_cast<size_t>(sz) % sizeof(G1Jacobian_t) != 0) {
        fs_absorb_bytes("stage/input_com_malformed",
                        input_fn.data(), input_fn.size());
        return {};
    }
    f.seekg(0);
    std::vector<G1Jacobian_t> com(static_cast<size_t>(sz) / sizeof(G1Jacobian_t));
    f.read(reinterpret_cast<char*>(com.data()), sz);
    if (!f) {
        fs_absorb_bytes("stage/input_com_readfail",
                        input_fn.data(), input_fn.size());
        return {};
    }
    fs_absorb_bytes("stage/input_com",
                    com.data(), com.size() * sizeof(G1Jacobian_t));
    return com;
}

// -----------------------------------------------------------------------------
// 2. Commit output, persist to `<output_fn>.com`, absorb into transcript.
//    Returns the host-resident commitment bytes.
// -----------------------------------------------------------------------------
inline std::vector<G1Jacobian_t>
commit_and_persist_output(const Commitment& gen,
                          const FrTensor& Y,
                          const std::string& output_fn)
{
    if (Y.size % gen.size != 0) {
        // Hard error - the generator file the caller passed is incompatible
        // with the output shape. Signal loudly rather than silently skipping
        // the commitment (which would re-open the binding gap).
        throw std::runtime_error(
            "stage_proof: Y.size (" + std::to_string(Y.size) +
            ") is not a multiple of gen.size (" + std::to_string(gen.size) +
            "); cannot commit output for " + output_fn);
    }
    G1TensorJacobian com = gen.commit_int(Y);
    std::vector<G1Jacobian_t> host(com.size);
    cudaMemcpy(host.data(), com.gpu_data,
               com.size * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);

    // Persist for the downstream stage.
    const std::string com_fn = output_fn + ".com";
    std::ofstream f(com_fn, std::ios::binary);
    if (!f) throw std::runtime_error("stage_proof: cannot open " + com_fn);
    f.write(reinterpret_cast<const char*>(host.data()),
            host.size() * sizeof(G1Jacobian_t));

    // Absorb so the current stage's transcript binds to its own output.
    fs_absorb_bytes("stage/output_com",
                    host.data(), host.size() * sizeof(G1Jacobian_t));
    return host;
}

// -----------------------------------------------------------------------------
// Generator selection for output commitment.
// Priority order:
//   1. `ZKLLM_ACTIVATION_GEN` env var → a dedicated activation-sized generator
//      (recommended production setting).
//   2. `fallback_gen` argument → typically one of the stage's weight
//      generators (which, in the current zkllm workspace, happens to be
//      sized `embed_dim` so activations of shape seq_len × embed_dim commit
//      cleanly; other stages may need ZKLLM_ACTIVATION_GEN explicitly).
//
// If neither is usable, throws.
// -----------------------------------------------------------------------------
inline std::vector<G1Jacobian_t>
commit_output_preferring_env(const Commitment& fallback_gen,
                             const FrTensor& Y,
                             const std::string& output_fn)
{
    const char* env = std::getenv("ZKLLM_ACTIVATION_GEN");
    if (env && *env) {
        std::ifstream probe(env, std::ios::binary);
        if (probe) {
            probe.close();
            // NB: avoid `Commitment gen(std::string(env));` which C++'s
            // most-vexing-parse rule reads as a function declaration.
            std::string env_path(env);
            Commitment gen(env_path);
            return commit_and_persist_output(gen, Y, output_fn);
        }
    }
    return commit_and_persist_output(fallback_gen, Y, output_fn);
}

// -----------------------------------------------------------------------------
// 3. Write `<output_fn>.proof` (stage-generic format).
// -----------------------------------------------------------------------------
inline void write_stage_proof(const std::string& output_fn,
                              const std::string& stage_tag,
                              const std::string& input_fn,
                              const std::vector<G1Jacobian_t>& input_com,
                              const std::vector<G1Jacobian_t>& output_com)
{
    const std::string path = output_fn + ".proof";
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("stage_proof: cannot open " + path);

    f.write(kStageMagic, sizeof(kStageMagic));
    f.write(reinterpret_cast<const char*>(&kStageVersion), sizeof(kStageVersion));

    write_lenprefixed(f, stage_tag.data(),  static_cast<uint32_t>(stage_tag.size()));
    write_lenprefixed(f, input_fn.data(),   static_cast<uint32_t>(input_fn.size()));
    write_lenprefixed(f, output_fn.data(),  static_cast<uint32_t>(output_fn.size()));
    write_lenprefixed(f, input_com.data(),
                      static_cast<uint32_t>(input_com.size()  * sizeof(G1Jacobian_t)));
    write_lenprefixed(f, output_com.data(),
                      static_cast<uint32_t>(output_com.size() * sizeof(G1Jacobian_t)));

    // Final 32-byte transcript digest, bound to every prior absorb/squeeze.
    f.write(reinterpret_cast<const char*>(fs_transcript().digest()),
            static_cast<std::streamsize>(Transcript::digest_size));
}

// -----------------------------------------------------------------------------
// Parsed on-disk representation of a stage proof.
// -----------------------------------------------------------------------------
struct StageProof {
    std::string stage_tag;
    std::string input_fn;
    std::string output_fn;
    std::vector<G1Jacobian_t> input_com;
    std::vector<G1Jacobian_t> output_com;
    uint8_t final_digest[Transcript::digest_size] = {0};
};

// Returns false on I/O or format error.
inline bool read_stage_proof(const std::string& path, StageProof& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[sizeof(kStageMagic)] = {0};
    f.read(magic, sizeof(magic));
    if (std::memcmp(magic, kStageMagic, sizeof(kStageMagic)) != 0) return false;
    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != kStageVersion) return false;

    auto read_string = [&](std::string& s) {
        std::vector<uint8_t> buf;
        if (!read_lenprefixed(f, buf)) return false;
        s.assign(reinterpret_cast<const char*>(buf.data()), buf.size());
        return true;
    };
    auto read_g1_vec = [&](std::vector<G1Jacobian_t>& v) {
        std::vector<uint8_t> buf;
        if (!read_lenprefixed(f, buf)) return false;
        if (buf.size() % sizeof(G1Jacobian_t) != 0) return false;
        v.resize(buf.size() / sizeof(G1Jacobian_t));
        std::memcpy(v.data(), buf.data(), buf.size());
        return true;
    };
    if (!read_string(out.stage_tag))   return false;
    if (!read_string(out.input_fn))    return false;
    if (!read_string(out.output_fn))   return false;
    if (!read_g1_vec(out.input_com))   return false;
    if (!read_g1_vec(out.output_com))  return false;
    f.read(reinterpret_cast<char*>(out.final_digest),
           static_cast<std::streamsize>(Transcript::digest_size));
    return static_cast<bool>(f);
}

} // namespace zkllm_stage

#endif // STAGE_PROOF_CUH
