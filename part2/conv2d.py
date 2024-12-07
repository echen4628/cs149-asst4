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

    out_channels_idx = nl.arange(nl.tile_size.pmax)[:,None,None]
    out_pool_height_idx = nl.arange(out_pool_height)[None,:,None]
    out_pool_width_idx = nl.arange(out_pool_width)[None,None,:]

    in_channels_idx = nl.arange(nl.tile_size.pmax)[:,None,None]
    out_height_idx = nl.arange(out_height)[None,:,None]
    out_width_idx = nl.arange(out_width)[None,None,:]

    # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
    TILE_WIDTH = 2*input_width

    # Maximum partition dimension of a tile
    TILE_128 = nl.tile_size.pmax  # 128

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for output_tile_idx in nl.affine_range(n_tiles_c_out):

            for output_row in nl.affine_range(out_height//2):
                
                partial_sum = nl.zeros((TILE_128, 2*out_width), nl.float32, buffer=nl.psum)
                for input_tile_idx in nl.affine_range(n_tiles_c_in):
                    W_tile = nl.ndarray((TILE_128, TILE_128, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            X_tile = nl.ndarray((TILE_128, 2, input_width), dtype=X.dtype, buffer=nl.sbuf)
                            X_tile[...] = nl.load(X[b, input_tile_idx*c_in_pmax:(input_tile_idx+1)*c_in_pmax, i+output_row*2:i+output_row*2+2,:])
                            X_tile_shifted = nl.copy(X_tile, dtype=X.dtype)[in_channels_idx, nl.arange(2)[None,:,None], out_width_idx+j]
                            X_tile_reshaped = nl.copy(X_tile_shifted, dtype=X.dtype).reshape((TILE_128, 2*out_width))
                            W_tile[...] = nl.load(W[output_tile_idx*c_in_pmax:(output_tile_idx+1)*c_in_pmax, input_tile_idx*c_in_pmax:(input_tile_idx+1)*c_in_pmax])
                            partial_sum += nl.matmul(nl.copy(W_tile[:,:,i,j]), X_tile_reshaped, transpose_x=False)
                complete_sum = nl.copy(partial_sum, dtype=X.dtype)
                current_bias = nl.load(bias[output_tile_idx*c_in_pmax:(output_tile_idx+1)*(c_in_pmax),], dtype=bias.dtype)
                # current_bias_broadcasted = nl.copy(current_bias).broadcast_to((c_in_pmax, 2*out_width))
                # temp_before_bias = nl.add(temp_before_bias,current_bias_broadcasted)
                complete_sum = nisa.tensor_scalar(complete_sum, np.add, current_bias)

                # X_out_tile_before_pooling_with_bias = (nl.load(bias[output_tile_idx*c_in_pmax:(output_tile_idx+1)*(c_in_pmax),])+X_out_tile_before_pooling)
                result_before_pooling = nl.copy(complete_sum).reshape((c_in_pmax, 2, out_width))
                # pdb.set_trace()

                # perform maxpooling
                if pool_size == 1:
                    nl.store(X_out[b, output_tile_idx*c_in_pmax:(output_tile_idx+1)*c_in_pmax, 2*output_row:2*output_row+2], value=result_before_pooling[...])
                elif pool_size == 2:
                    max_between_rows = nl.max(result_before_pooling, axis=1)
                    max_between_cols = nl.copy(max_between_rows).reshape((c_in_pmax, out_width/pool_size, pool_size))
                    max_between_cols = nl.max(max_between_cols, axis=2)
                    result_after_pooling = nl.copy(max_between_cols).reshape((c_in_pmax, 1, out_width/pool_size))

                    # temp1 = nl.copy(result_before_pooling)
                    # temp2 = nl.max(temp1, axis=1)
                    # temp3 = nl.copy(temp2).reshape((c_in_pmax, out_width/pool_size, pool_size))
                    # temp4 = nl.max(temp3, axis=2)
                    # temp5 = nl.copy(temp4).reshape((c_in_pmax, 1, out_width/pool_size))

                    nl.store(X_out[b, output_tile_idx*c_in_pmax:(output_tile_idx+1)*c_in_pmax, output_row:output_row+1], value=result_after_pooling[...])
    return X_out
