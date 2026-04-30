#include "zkrelu.cuh" 
#include <stdexcept>

zkReLU::zkReLU(uint scaling_factor): scaling_factor(scaling_factor), tl_rem(-static_cast<int>(scaling_factor>>1), scaling_factor), sign_tensor_ptr(nullptr), abs_tensor_ptr(nullptr), rem_tensor_ptr(nullptr), m_tensor_ptr(nullptr)
{
}

// void decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem, FrTensor& rem_ind);
KERNEL void zkrelu_decomp_kernel(Fr_t* X_ptr, Fr_t* sign_ptr, Fr_t* abs_ptr, Fr_t* rem_ptr, Fr_t* res_ptr, long scaling_factor, uint N)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {   
        long hsf = scaling_factor >> 1;
        long x = scalar_to_long(X_ptr[tid]);
        long temp = (x + hsf) % scaling_factor;
        long x_rem = temp < 0 ? temp + scaling_factor : temp;
        x_rem -= hsf;
        long x_rescaled = (x - x_rem) / scaling_factor;


        bool pos = x_rescaled >= 0;

        sign_ptr[tid] = {static_cast<uint>(pos), 0, 0, 0, 0, 0, 0, 0};
        abs_ptr[tid] = pos? long_to_scalar(x_rescaled) : long_to_scalar(-x_rescaled);
        rem_ptr[tid] = long_to_scalar(x_rem);
        res_ptr[tid] = pos? long_to_scalar(x_rescaled) : blstrs__scalar__Scalar_ZERO;
    }
}

FrTensor zkReLU::decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem)
{
    uint N = X.size;
    FrTensor res(N);
    uint block_size = 256;
    uint grid_size = (N + block_size - 1) / block_size;
    zkrelu_decomp_kernel<<<grid_size, block_size>>>(X.gpu_data, sign.gpu_data, abs.gpu_data, rem.gpu_data, res.gpu_data, static_cast<long>(scaling_factor), N);
    cudaDeviceSynchronize();
    return res;
}

FrTensor zkReLU::operator()(const FrTensor& X)
{
    if (sign_tensor_ptr) delete sign_tensor_ptr;
    sign_tensor_ptr = new FrTensor(X.size);
    if (abs_tensor_ptr) delete abs_tensor_ptr;
    abs_tensor_ptr = new FrTensor(X.size);
    if (rem_tensor_ptr) delete rem_tensor_ptr;
    rem_tensor_ptr = new FrTensor(X.size);
    if (m_tensor_ptr) delete m_tensor_ptr;
    // m_tensor_ptr = new FrTensor(tl_rem.table.size);

    FrTensor res = decomp(X, *sign_tensor_ptr, *abs_tensor_ptr, *rem_tensor_ptr);
    m_tensor_ptr = new FrTensor(tl_rem.prep(*rem_tensor_ptr));
    return res;
}

void zkReLU::prove(const FrTensor& Z, const FrTensor& A)
{
    // This method is not called anywhere in the current LLaMA pipeline (which
    // uses SwiGLU / SiLU via tLookupRangeMapping in ffn.cu), and no validated
    // end-to-end lookup-proof implementation exists. Rather than ship a
    // fabricated one, we fail fast so any future caller is forced to implement
    // the sign/|X|/remainder lookup proof.
    //
    // Protocol sketch for a future implementer:
    //   Decomposition already computed in operator(): X = sign * (|X| * s + r)
    //     where s = scaling_factor, |sign| ∈ {0, 1}, |X|, r ∈ tabled ranges.
    //   Required proofs:
    //     (1) sign_i * (sign_i - 1) = 0          (boolean via sumcheck)
    //     (2) Z_i = sign_i * (|X|_i * s + r_i)   (hadamard + rescale sumcheck)
    //     (3) A_i = sign_i * |X|_i               (hadamard sumcheck)
    //     (4) r_i ∈ [-s/2, s/2)                  (tLookup tl_rem, already set up)
    //     (5) |X|_i ∈ [0, 2^bits)                (tLookup range proof)
    //   All challenges must come from fs_challenge_vec / fs_challenge_fr with
    //   Z and A commitments absorbed first.
    throw std::runtime_error(
        "zkReLU::prove() is not implemented in this fork. "
        "The LLaMA pipeline does not currently invoke this method; if you "
        "need ReLU proofs, implement the 5-step protocol sketched above in "
        "zkrelu.cu.");
}

zkReLU::~zkReLU()
{
    if (sign_tensor_ptr) delete sign_tensor_ptr;
    if (abs_tensor_ptr) delete abs_tensor_ptr;
    if (rem_tensor_ptr) delete rem_tensor_ptr;
    if (m_tensor_ptr) delete m_tensor_ptr;
}