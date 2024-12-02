import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
import pdb as pdb


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

# @nki.jit
# def fused_conv2d_maxpool(X, W, bias, pool_size=1):
#     # print("hello")
#     # print(X)
#     # print(W)
#     pdb.set_trace()
#     batch_size, in_channels, input_height, input_width = X.shape
#     out_channels, in_channels_, filter_height, filter_width = W.shape
#     # W_reshape = W.transpose((2,3,0,1))
#     out_channels_ = bias.shape[0]

#     assert (
#         in_channels_ == in_channels and out_channels_ == out_channels
#     ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

#     out_height = input_height - filter_height + 1
#     out_width = input_width - filter_width + 1

#     out_pool_height = out_height // pool_size
#     out_pool_width = out_width // pool_size
    
#     # Can assume multiple of 128 to avoid using mask
#     assert in_channels % 128 == 0

#     # Can assume one PSUM bank can at least fit one row of the pixels
#     assert nl.tile_size.gemm_moving_fmax >= out_width

#     # Initialize output array
#     # X_out = nl.ndarray(
#     #     shape=(batch_size, out_channels, out_pool_height, out_pool_width),
#     #     dtype=X.dtype,
#     #     buffer=nl.hbm,
#     # )
#     pdb.set_trace()
#     X_out = nl.ndarray(
#         shape=(batch_size, out_channels, out_height, out_width),
#         dtype=X.dtype,
#         buffer=nl.hbm,
#     )

#     # Various tiling dimensions (You may want to define more of them)
#     c_in_pmax = nl.tile_size.pmax
#     n_tiles_c_in = in_channels // c_in_pmax


#     # Process the images in batches
#     W_loaded= nl.load(W)
#     for b in nl.affine_range(batch_size):
#         # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
#         # and store the result in X_out[b]
#         X_out_b = nl.load(X_out[b]).reshape((out_channels, out_height*out_width))
#         for i in nl.affine_range(filter_height):
#             for j in nl.affine_range(filter_width):
#                 pdb.set_trace()
#                 # X[b] times W
#                 # X[b]_reshape to have shape H*W rows by input_channels
#                 # W[:, :, i,j] have shape output_channels rows and  input_channels cols
#                 X_b = nl.load(X[b])
#                 in_channels_idx = nl.arange(in_channels)[:,None,None]
#                 out_height_idx = nl.arange(out_height)[None,:,None]
#                 out_width_idx = nl.arange(out_width)[None,None,:]
#                 X_temp = nl.ndarray((in_channels, out_height, out_width), dtype=X_b.dtype)
#                 X_temp[in_channels_idx, out_height_idx, out_width_idx] = X_b[in_channels_idx, out_height_idx+i, out_width_idx+j]
#                 # X_b[in_channels_idx, out_height_idx, out_width_idx]
#                 X_reshape = X_temp.reshape((in_channels, out_height*out_width)) # 128 by 512, use this as rhs
#                 # X_reshape_shifted = X_reshape[in_channels_idx, out_height_idx*input_width+out_width_idx]
#                 # W_i_j_T = nki.isa.nc_transpose(W_i_j)
#                 X_out_b += nl.matmul(W_loaded[:,:,i,j], X_reshape, transpose_x=False)
#         pdb.set_trace()
#         nl.store(X_out[b], value=X_out_b.reshape((out_channels, out_height, out_width)))
#         # X_out[b] = nl.copy(X_out_b)
#     pdb.set_trace()
#     # X_out = nl.load(X_out)
#     # nl.store(out, value=X_out.reshape((batch_size, out_channels, out_height, out_width)))
#     # return X_out
#     return X_out


# @nki.jit
# def fused_conv2d_maxpool(X, W, bias, pool_size=1):
#     batch_size, in_channels, input_height, input_width = X.shape
#     out_channels, in_channels_, filter_height, filter_width = W.shape
#     out_channels_ = bias.shape[0]

#     assert (
#         in_channels_ == in_channels and out_channels_ == out_channels
#     ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

#     out_height = input_height - filter_height + 1
#     out_width = input_width - filter_width + 1

#     out_pool_height = out_height // pool_size
#     out_pool_width = out_width // pool_size
    
#     # Can assume multiple of 128 to avoid using mask
#     assert in_channels % 128 == 0

#     # Can assume one PSUM bank can at least fit one row of the pixels
#     assert nl.tile_size.gemm_moving_fmax >= out_width

#     # Initialize output array
#     X_out = nl.ndarray(
#         shape=(batch_size, out_channels, out_pool_height, out_pool_width),
#         dtype=X.dtype,
#         buffer=nl.hbm,
#     )
#     # pdb.set_trace()
#     X_out_before_pooling = nl.ndarray(
#         shape=(batch_size, out_channels, out_height, out_width),
#         dtype=X.dtype,
#         buffer=nl.hbm,
#     )

#     # Various tiling dimensions (You may want to define more of them)
#     c_in_pmax = nl.tile_size.pmax
#     n_tiles_c_in = in_channels // c_in_pmax


#     # Process the images in batches
#     W_loaded= nl.load(W)
#     for b in nl.affine_range(batch_size):
#         pdb.set_trace()
#         print("awef")
#         # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
#         # and store the result in X_out[b]
#         X_out_before_pooling_b = nl.load(X_out_before_pooling[b]).reshape((out_channels, out_height*out_width))
#         X_out_after_pooling_b = nl.load(X_out[b])
#         for i in nl.affine_range(filter_height):
#             for j in nl.affine_range(filter_width):
#                 print("hi")
#                 X_b = nl.load(X[b])
#                 in_channels_idx = nl.arange(in_channels)[:,None,None]
#                 out_height_idx = nl.arange(out_height)[None,:,None]
#                 out_width_idx = nl.arange(out_width)[None,None,:]
#                 X_temp = nl.ndarray((in_channels, out_height, out_width), dtype=X_b.dtype)
#                 X_temp[in_channels_idx, out_height_idx, out_width_idx] = X_b[in_channels_idx, out_height_idx+i, out_width_idx+j]
#                 X_reshape = X_temp.reshape((in_channels, out_height*out_width)) # 128 by 512, use this as rh
#                 X_out_before_pooling_b += nl.matmul(W_loaded[:,:,i,j], X_reshape, transpose_x=False)
#         pool_channels_idx = nl.arange(in_channels)[:,None,None]
#         pool_height_idx = nl.arange(out_pool_height)[None, :, None]
#         pool_width_idx = nl.arange(out_pool_width)[None, None, :]
#         X_out_after_pooling_b = X_out_before_pooling_b[pool_channels_idx, pool_height_idx*out_width*pool_size+pool_width_idx*pool_size]
#         nl.store(X_out[b], value=X_out_after_pooling_b)
#     return X_out

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_in_pmax
    pdb.set_trace()

    out_channels_idx = nl.arange(c_in_pmax)[:,None,None]
    out_pool_height_idx = nl.arange(out_pool_height)[None,:,None]
    out_pool_width_idx = nl.arange(out_pool_width)[None,None,:]

    in_channels_idx = nl.arange(c_in_pmax)[:,None,None]
    out_height_idx = nl.arange(out_height)[None,:,None]
    out_width_idx = nl.arange(out_width)[None,None,:]

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for tile_c_out in nl.affine_range(n_tiles_c_out):
            X_out_before_pooling = nl.ndarray(shape=(c_in_pmax, out_height*out_width),
                                    dtype=X.dtype,
                                    buffer=nl.sbuf,) # change to 128
            for tile_c_in in nl.affine_range(n_tiles_c_in):
                

                    
                for i in nl.affine_range(filter_height):
                    for j in nl.affine_range(filter_width):
                        # only load in 128 in channels
                        X_b = nl.load(X[b,c_in_pmax*img_tile_c_in:c_in_pmax*(img_tile_c_in+1)]) # pick first 128
                        W_loaded = nl.load(W)
                        X_temp = nl.ndarray((c_in_pmax, out_height, out_width), dtype=X_b.dtype) # change to 128
                        X_temp[in_channels_idx, out_height_idx, out_width_idx] = X_b[in_channels_idx, out_height_idx+i, out_width_idx+j]
                        X_reshape = X_temp.reshape((c_in_pmax, out_height*out_width)) # 128 by 420, use this as rh
                        X_out_before_pooling += nl.matmul(nl.transpose(W_loaded[:,:,i,j]), X_reshape, transpose_x=True) # 128 by 420
                nl.store(X_out[b, c_in_pmax*img_tile_c_in:c_in_pmax*(img_tile_c_in+1)], value=X_out_before_pooling.reshape((c_in_pmax, out_height, out_width))[out_channels_idx, out_pool_height_idx, out_pool_width_idx])
    return X_out


def nki_matmul_tiled_(lhsT, rhs, result):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner"""

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

  # Maximum partition dimension of a tile
  TILE_K = nl.tile_size.pmax  # 128

  # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        lhsT_tile[...] = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K,
                                      m * TILE_M:(m + 1) * TILE_M])
        rhs_tile[...] = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                    n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
               value=res_sb)