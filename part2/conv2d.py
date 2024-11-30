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

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):
    # print("hello")
    # print(X)
    # print(W)
    pdb.set_trace()
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    # W_reshape = W.transpose((2,3,0,1))
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
    # X_out = nl.ndarray(
    #     shape=(batch_size, out_channels, out_pool_height, out_pool_width),
    #     dtype=X.dtype,
    #     buffer=nl.hbm,
    # )
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height*out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax


    # Process the images in batches
    W_loaded= nl.load(W)
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]
        X_out_b = nl.load(X_out[b])
        for i in nl.affine_range(filter_height):
            for j in nl.affine_range(filter_width):
                pdb.set_trace()
                # X[b] times W
                # X[b]_reshape to have shape H*W rows by input_channels
                # W[:, :, i,j] have shape output_channels rows and  input_channels cols
                X_b = nl.load(X[b])
                in_channels_idx = nl.arange(in_channels)[:,None,None]
                out_height_idx = nl.arange(out_height)[None,:,None]
                out_width_idx = nl.arange(out_width)[None,None,:]
                X_temp = nl.ndarray((in_channels, out_height, out_width), dtype=X_b.dtype)
                X_temp[in_channels_idx, out_height_idx, out_width_idx] = X_b[in_channels_idx, out_height_idx+i, out_width_idx+j]
                # X_b[in_channels_idx, out_height_idx, out_width_idx]
                X_reshape = X_temp.reshape((in_channels, out_height*out_width)) # 128 by 512, use this as rhs
                # X_reshape_shifted = X_reshape[in_channels_idx, out_height_idx*input_width+out_width_idx]
                # W_i_j_T = nki.isa.nc_transpose(W_i_j)
                X_out_b += nl.matmul(W_loaded[:,:,i,j], X_reshape, transpose_x=False)
        pdb.set_trace()
        X_out[b] = nl.copy(X_out_b)

                




        # # reshape X[b]
        # pdb.set_trace()
        # X_b = nl.load(X[b])
        # X_b_flatten = X_b.reshape((in_channels, input_height*input_width))
        # X_b_transposed = nl.ndarray(shape=(input_height*input_width, in_channels),dtype=X.dtype,buffer=nl.hbm)
        # X_b_i_p_a = nl.arange(in_channels)[:, None]
        # for rows in nl.affine_range((input_height*input_width+128-1) // 128):
        #     X_b_i_f_a = nl.arange(128*(rows+1))[None, :]
        #     # assuming all the in channels need to be used
        #     X_b_transposed[128*rows:min(128*(rows+1), input_height*input_width), :] = nki.isa.nc_transpose(X_b_flatten, mask=((X_b_i_p_a < 128) & (X_b_i_f_a >= 128*(rows))))

        # continue


    return X_out

