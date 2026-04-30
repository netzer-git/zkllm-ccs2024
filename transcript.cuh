#ifndef TRANSCRIPT_CUH
#define TRANSCRIPT_CUH

// =============================================================================
// Fiat-Shamir Transcript for zkLLM (host-side).
//
// All prover/verifier challenge draws MUST go through a single Transcript
// instance shared between the prover and the verifier. Challenges produced
// here are deterministic functions of every value previously absorbed, so any
// prover attempt to pick a challenge before committing to the values it binds
// will cause verification to fail.
//
// Design: simple sponge over SHA3-256 (Keccak-f[1600], 136-byte rate). We
// maintain a running 32-byte chaining value `state_`; every absorb updates it
// via `state_ = SHA3_256(state_ || domain_tag || label_len || label || data_len
// || data)`, every challenge squeezes via `state_ = SHA3_256(state_ ||
// "challenge" || label || out_len)` and returns the first `out_len` bytes.
// This is not a true duplex construction but provides collision resistance and
// the required "commit before challenge" binding.
//
// Pure-host C++; safe to include from both .cu and .cpp.
// =============================================================================

#include <cstdint>
#include <cstring>
#include <cstddef>
#include <string>
#include <vector>
#include "fr-tensor.cuh"   // for Fr_t (struct with uint32_t val[8])
#include "g1-tensor.cuh"   // for G1Affine_t / G1Jacobian_t (used by absorb_g1)

// -----------------------------------------------------------------------------
// Compact Keccak-f[1600] / SHA3-256 implementation.
// Adapted from Markku-Juhani O. Saarinen's public-domain tiny_sha3
// (https://github.com/mjosaarinen/tiny_sha3), rewritten as header-only inline.
// -----------------------------------------------------------------------------
namespace zkllm_fs {

static inline uint64_t rotl64(uint64_t x, unsigned n) { return (x << n) | (x >> (64 - n)); }

static inline void keccakf(uint64_t st[25]) {
    static const uint64_t rc[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
    static const unsigned r[24] = {
         1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
        27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44};
    static const unsigned p[24] = {
        10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
        15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1};

    uint64_t bc[5], t;
    for (int round = 0; round < 24; ++round) {
        // Theta
        for (int i = 0; i < 5; ++i) bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
        for (int i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        // Rho + Pi
        t = st[1];
        for (int i = 0; i < 24; ++i) {
            int j = p[i];
            bc[0] = st[j];
            st[j] = rotl64(t, r[i]);
            t = bc[0];
        }
        // Chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; ++i) bc[i] = st[j + i];
            for (int i = 0; i < 5; ++i)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        // Iota
        st[0] ^= rc[round];
    }
}

// SHA3-256: rate = 136 bytes, capacity = 64 bytes, output = 32 bytes, domain = 0x06.
static inline void sha3_256(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint64_t st[25] = {0};
    uint8_t* st_bytes = reinterpret_cast<uint8_t*>(st);
    const size_t rate = 136;
    size_t off = 0;
    while (len >= rate) {
        for (size_t i = 0; i < rate; ++i) st_bytes[i] ^= data[off + i];
        keccakf(st);
        off += rate;
        len -= rate;
    }
    // Pad: 0x06 ... 0x80 (SHA-3 domain separation).
    uint8_t buf[rate];
    std::memset(buf, 0, rate);
    if (len) std::memcpy(buf, data + off, len);
    buf[len] = 0x06;
    buf[rate - 1] |= 0x80;
    for (size_t i = 0; i < rate; ++i) st_bytes[i] ^= buf[i];
    keccakf(st);
    std::memcpy(out, st_bytes, 32);
}

} // namespace zkllm_fs

// -----------------------------------------------------------------------------
// Transcript class.
// -----------------------------------------------------------------------------
class Transcript {
public:
    // A fresh transcript is seeded with a domain-separator tag so transcripts
    // from different protocols cannot be mixed.
    explicit Transcript(const std::string& protocol_tag = "zkLLM/v1") {
        state_.assign(32, 0);
        absorb("protocol", protocol_tag.data(), protocol_tag.size());
    }

    // Generic absorb: chain in (label, data).
    void absorb(const char* label, const void* data, size_t len) {
        std::vector<uint8_t> buf;
        buf.reserve(32 + 8 + std::strlen(label) + 8 + len);
        append_bytes(buf, state_.data(), 32);
        append_label(buf, "absorb");
        append_label(buf, label);
        append_u64(buf, static_cast<uint64_t>(len));
        append_bytes(buf, data, len);
        uint8_t out[32];
        zkllm_fs::sha3_256(buf.data(), buf.size(), out);
        std::memcpy(state_.data(), out, 32);
    }

    // Convenience overloads.
    void absorb_fr(const char* label, const Fr_t& x) {
        absorb(label, x.val, sizeof(x.val));
    }

    void absorb_fr_vec(const char* label, const std::vector<Fr_t>& v) {
        if (v.empty()) { absorb(label, nullptr, 0); return; }
        absorb(label, v.data(), v.size() * sizeof(Fr_t));
    }

    // Absorb a G1 group element (commitment). Used by code-inspector finding
    // #2 to bind verifier challenges to all prior prover-sent commitments.
    void absorb_g1(const char* label, const G1Affine_t& p) {
        absorb(label, &p, sizeof(p));
    }

    void absorb_g1_vec(const char* label, const std::vector<G1Affine_t>& v) {
        if (v.empty()) { absorb(label, nullptr, 0); return; }
        absorb(label, v.data(), v.size() * sizeof(G1Affine_t));
    }

    // Squeeze a single field challenge bound to every prior absorb.
    Fr_t challenge_fr(const char* label) {
        uint8_t out[32];
        squeeze_bytes(label, out, 32);
        return bytes_to_fr(out);
    }

    // Squeeze a vector of field challenges. Equivalent to n independent calls
    // with labels "label/0", "label/1", ..., but more efficient.
    std::vector<Fr_t> challenge_vec(const char* label, uint32_t n) {
        std::vector<Fr_t> out(n);
        // Derive a 32-byte seed bound to (state, label, n), then expand it.
        uint8_t seed[32];
        {
            std::vector<uint8_t> buf;
            buf.reserve(32 + 16 + std::strlen(label) + 8);
            append_bytes(buf, state_.data(), 32);
            append_label(buf, "challenge_vec");
            append_label(buf, label);
            append_u64(buf, static_cast<uint64_t>(n));
            zkllm_fs::sha3_256(buf.data(), buf.size(), seed);
        }
        // Update state to the seed so subsequent absorbs/challenges chain.
        std::memcpy(state_.data(), seed, 32);
        // Expand seed into n field elements via SHA3-256(seed || index).
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t chunk[32];
            uint8_t buf[32 + 8];
            std::memcpy(buf, seed, 32);
            uint64_t idx = i;
            for (int b = 0; b < 8; ++b) buf[32 + b] = static_cast<uint8_t>((idx >> (8 * b)) & 0xff);
            zkllm_fs::sha3_256(buf, sizeof(buf), chunk);
            out[i] = bytes_to_fr(chunk);
        }
        return out;
    }

    // Read-only view of the current 32-byte chaining state, used by
    // stage_proof.cuh to persist a cryptographic fingerprint of every public
    // input and prover message seen so far.
    // Callers must NOT mutate the returned pointer.
    const uint8_t* digest() const { return state_.data(); }
    static constexpr size_t digest_size = 32;

private:
    // Internal state: 32 bytes chaining value.
    std::vector<uint8_t> state_;

    static void append_bytes(std::vector<uint8_t>& buf, const void* data, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        buf.insert(buf.end(), p, p + len);
    }

    static void append_u64(std::vector<uint8_t>& buf, uint64_t x) {
        for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xff));
    }

    static void append_label(std::vector<uint8_t>& buf, const char* label) {
        size_t n = std::strlen(label);
        append_u64(buf, static_cast<uint64_t>(n));
        append_bytes(buf, label, n);
    }

    void squeeze_bytes(const char* label, uint8_t* out, size_t len) {
        std::vector<uint8_t> buf;
        buf.reserve(32 + 16 + std::strlen(label) + 8);
        append_bytes(buf, state_.data(), 32);
        append_label(buf, "challenge");
        append_label(buf, label);
        append_u64(buf, static_cast<uint64_t>(len));
        uint8_t hash[32];
        zkllm_fs::sha3_256(buf.data(), buf.size(), hash);
        // Update state and emit output. We want |out| <= 32 here.
        std::memcpy(state_.data(), hash, 32);
        if (len > 32) len = 32; // not used for len > 32 in this codebase
        std::memcpy(out, hash, len);
    }

    // Convert 32 bytes into an Fr_t (struct with uint32_t val[8]).
    // Matches the existing `random_vec` reduction: the top limb is taken mod
    // 1944954707 (= 0x73eda753, the top 32-bit limb of the BLS12-381 scalar
    // field modulus r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001).
    // Remaining limbs use the full uint32 range; the final value is reduced
    // mod r at point of use (Montgomery form), so the tiny residual bias is
    // not a soundness issue.
    static Fr_t bytes_to_fr(const uint8_t b[32]) {
        Fr_t x;
        for (int i = 0; i < 8; ++i) {
            x.val[i] = (static_cast<uint32_t>(b[4 * i])       )
                     | (static_cast<uint32_t>(b[4 * i + 1]) <<  8)
                     | (static_cast<uint32_t>(b[4 * i + 2]) << 16)
                     | (static_cast<uint32_t>(b[4 * i + 3]) << 24);
        }
        x.val[7] = x.val[7] % 1944954707u;
        return x;
    }
};

// -----------------------------------------------------------------------------
// Process-global transcript accessor.
//
// A single Transcript instance is shared across every stage of the zkLLM
// pipeline (ppgen, commit-param, self-attn, ffn, rmsnorm, skip-connection,
// main verifier). This is the lightest-touch way to enforce Fiat-Shamir
// without threading a Transcript& through every existing prove()/verify()
// signature. Each stage's main() MUST:
//   1. Seed it by calling fs_transcript_init(stage_tag, ...).
//   2. Absorb every commitment / public input it sees (fs_absorb_* helpers).
//   3. Only then draw challenges via fs_challenge_vec / fs_challenge_fr.
//
// Defined in fr-tensor.cu so it's always linked with the core field library.
// -----------------------------------------------------------------------------
Transcript& fs_transcript();
void        fs_transcript_init(const std::string& stage_tag);

inline void fs_absorb_fr    (const char* label, const Fr_t& x)                 { fs_transcript().absorb_fr(label, x); }
inline void fs_absorb_fr_vec(const char* label, const std::vector<Fr_t>& v)    { fs_transcript().absorb_fr_vec(label, v); }
inline void fs_absorb_g1    (const char* label, const G1Affine_t& p)           { fs_transcript().absorb_g1(label, p); }
inline void fs_absorb_g1_vec(const char* label, const std::vector<G1Affine_t>& v) { fs_transcript().absorb_g1_vec(label, v); }
inline void fs_absorb_bytes (const char* label, const void* data, size_t len)  { fs_transcript().absorb(label, data, len); }

#endif // TRANSCRIPT_CUH
