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

#     # Various tiling dimensions (You may want to define more of them)
#     c_in_pmax = nl.tile_size.pmax
#     n_tiles_c_in = in_channels // c_in_pmax
#     n_tiles_c_out = out_channels // c_in_pmax

#     out_channels_idx = nl.arange(c_in_pmax)[:,None,None]
#     out_pool_height_idx = nl.arange(out_pool_height)[None,:,None]
#     out_pool_width_idx = nl.arange(out_pool_width)[None,None,:]

#     in_channels_idx = nl.arange(c_in_pmax)[:,None,None]
#     out_height_idx = nl.arange(out_height)[None,:,None]
#     out_width_idx = nl.arange(out_width)[None,None,:]

#     # Process the images in batches
#     for b in nl.affine_range(batch_size):
#         for out_tile in nl.affine_range(n_tiles_c_out):
#             for i in nl.affine_range(filter_height):
#                 for j in nl.affine_range(filter_width):
#                     X_out_before_pooling = nl.ndarray(shape=(c_in_pmax, out_height*out_width),
#                                                         dtype=X.dtype,
#                                                         buffer=nl.sbuf)
#                     for in_tile in nl.affine_range(n_tiles_c_in):

            
#                 # for i in nl.affine_range(filter_height):
#                 #     for j in nl.affine_range(filter_width):
#                         X_b = nl.load(X[b, c_in_pmax*in_tile:c_in_pmax*(in_tile+1)])
#                         W_loaded = nl.load(W[c_in_pmax*out_tile:c_in_pmax*(out_tile+1), c_in_pmax*in_tile:c_in_pmax*(in_tile+1)]) 
#                         X_temp = nl.ndarray((c_in_pmax, out_height, out_width), dtype=X_b.dtype)
#                         # edit below
#                         X_temp[in_channels_idx, out_height_idx, out_width_idx] = X_b[in_channels_idx, out_height_idx+i, out_width_idx+j]
#                         X_reshape = X_temp.reshape((c_in_pmax, out_height*out_width)) # 128 by 420, use this as rh
#                         pdb.set_trace()
#                         X_out_before_pooling += nl.matmul(W_loaded[:,:,i,j], X_reshape, transpose_x=False) # 128 by 420
#             pdb.set_trace()
#             nl.store(X_out[b, c_in_pmax*out_tile:c_in_pmax*(out_tile+1)], value=X_out_before_pooling.reshape((c_in_pmax, out_height, out_width))[out_channels_idx, out_pool_height_idx, out_pool_width_idx])
#             # nl.store(X_out[b], value=X_b[out_channels_idx, out_pool_height_idx, out_pool_width_idx])

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

#     # Various tiling dimensions (You may want to define more of them)
#     c_in_pmax = nl.tile_size.pmax
#     n_tiles_c_in = in_channels // c_in_pmax
#     n_tiles_c_out = out_channels // c_in_pmax

#     out_channels_idx = nl.arange(nl.tile_size.pmax)[:,None,None]
#     out_pool_height_idx = nl.arange(out_pool_height)[None,:,None]
#     out_pool_width_idx = nl.arange(out_pool_width)[None,None,:]

#     in_channels_idx = nl.arange(nl.tile_size.pmax)[:,None,None]
#     out_height_idx = nl.arange(out_height)[None,:,None]
#     out_width_idx = nl.arange(out_width)[None,None,:]

#     # Process the images in batches
#     for b in nl.affine_range(batch_size):
#         for out_tile in nl.affine_range(n_tiles_c_out):
#             X_out_tile_before_pooling = nl.zeros((c_in_pmax, out_height*out_width),  X_out.dtype)
#             X_out_tile_after_pooling = nl.zeros((c_in_pmax, out_pool_height, out_pool_width), X_out.dtype)
#             for i in nl.affine_range(filter_height):
#                 for j in nl.affine_range(filter_width):
#                     X_b_tile = nl.tile() # size c_in_max by 512
#                     for in_tile in nl.affine_range(n_tiles_c_in):
#                         for k in nl.affine_range():
#                         X_b = nl.load(X[b, c_in_pmax*in_tile:c_in_pmax*(in_tile+1)])
#                         W_loaded = nl.load(W[c_in_pmax*out_tile:c_in_pmax*(out_tile+1), c_in_pmax*in_tile:c_in_pmax*(in_tile+1)])
                
#                         # shifted X
#                         X_temp = nl.ndarray((c_in_pmax, out_height, out_width), X_out.dtype)
#                         X_temp[in_channels_idx, out_height_idx, out_width_idx] = X_b[in_channels_idx, out_height_idx+i, out_width_idx+j]
#                         X_reshape = X_temp.reshape((c_in_pmax, out_height*out_width))
#                         X_out_tile_before_pooling += nl.matmul(W_loaded[:,:,i,j], X_reshape, transpose_x=False)
#                 # temp_output_tile_accum += X_b[out_channels_idx, out_height_idx, out_width_idx]
#             X_out_tile_after_pooling = X_out_tile_before_pooling.reshape((c_in_pmax, out_height, out_width))[out_channels_idx, out_pool_height_idx, out_pool_width_idx]
#             nl.store(X_out[b, c_in_pmax*out_tile:c_in_pmax*(out_tile+1)], value=X_out_tile_after_pooling)


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

#     # Various tiling dimensions (You may want to define more of them)
#     c_in_pmax = nl.tile_size.pmax
#     n_tiles_c_in = in_channels // c_in_pmax
#     n_tiles_c_out = out_channels // c_in_pmax

#     out_channels_idx = nl.arange(nl.tile_size.pmax)[:,None,None]
#     out_pool_height_idx = nl.arange(out_pool_height)[None,:,None]
#     out_pool_width_idx = nl.arange(out_pool_width)[None,None,:]

#     in_channels_idx = nl.arange(nl.tile_size.pmax)[:,None,None]
#     out_height_idx = nl.arange(out_height)[None,:,None]
#     out_width_idx = nl.arange(out_width)[None,None,:]

#     # Process the images in batches
#     for b in nl.affine_range(batch_size):
#         for output_tile_idx in nl.affine_range(n_tiles_c_out):
#             for input_row in nl.affine_range(out_pool_height//2):
#                     # create the accumulator in psum
#                     X_out_tile_before_pooling = nl.zeros((c_in_pmax, 2, out_width),  X_out.dtype, buffer=nl.psum)
#                     for i in nl.affine_range(filter_height):
#                         for input_tile_idx in nl.affine_range(n_tiles_c_in):
#                             for j in nl.affine_range(filter_width):
#                                 # shift row
#                                 shifted_row = 2*input_row*pool_size+i
#                                 X_rhs_tile = nl.load(X[b,c_in_pmax*input_tile_idx:c_in_pmax*(input_tile_idx+1), shifted_row:shifted_row+2, :])
                                
#                                 # shift the columns
#                                 X_rhs_tile_shifted = X_rhs_tile[in_channels_idx, nl.arange(2)[None,:,None], out_width_idx+j]
#                                 W_tile = nl.load(W[c_in_pmax*output_tile_idx:c_in_pmax*(output_tile_idx+1), c_in_pmax*input_tile_idx:c_in_pmax*(input_tile_idx+1)])
                                
#                                 # do matmul with W
#                                 X_out_tile_before_pooling += nl.matmul(W_tile[:,:,i,j], X_rhs_tile_shifted, transpose_x=False)
#                     # add bias
#                     X_out_tile_before_pooling_with_bias = (nl.load(bias[output_tile_idx*c_in_pmax:(output_tile_idx+1)*(c_in_pmax),])+X_out_tile_before_pooling)
                    
#                     # perform maxpooling
#                     temp1 = nl.copy(X_out_tile_before_pooling_with_bias.reshape((c_in_pmax, pool_size, 2/pool_size, out_width)))
#                     temp2 = nl.max(temp1, axis=1)
#                     temp3 = nl.copy(temp2.reshape((c_in_pmax, 2/pool_size, pool_size, out_width/pool_size)))
#                     X_out_tile_after_pooling = nl.max(temp3, axis=2)
#                     # X_out_tile_after_pooling = nl.max(nl.max(X_out_tile_before_pooling_with_bias.reshape((c_in_pmax, pool_size, 2/pool_size, out_width)), axis=1).reshape((c_in_pmax, 2/pool_size, pool_size, out_width/pool_size)), axis=2)
#                     nl.store(X_out[b, c_in_pmax*output_tile_idx:c_in_pmax*(output_tile_idx+1), 2*input_row:(2*input_row+2)], value=X_out_tile_after_pooling)
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

#     # Various tiling dimensions (You may want to define more of them)
#     c_in_pmax = nl.tile_size.pmax
#     n_tiles_c_in = in_channels // c_in_pmax
#     n_tiles_c_out = out_channels // c_in_pmax

#     out_channels_idx = nl.arange(nl.tile_size.pmax)[:,None,None]
#     out_pool_height_idx = nl.arange(out_pool_height)[None,:,None]
#     out_pool_width_idx = nl.arange(out_pool_width)[None,None,:]

#     in_channels_idx = nl.arange(nl.tile_size.pmax)[:,None,None]
#     out_height_idx = nl.arange(out_height)[None,:,None]
#     out_width_idx = nl.arange(out_width)[None,None,:]

#     # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
#     TILE_WIDTH = 2*input_width

#     # Maximum partition dimension of a tile
#     TILE_128 = nl.tile_size.pmax  # 128

#     # Process the images in batches
#     for b in nl.affine_range(batch_size):
#         for output_tile_idx in nl.affine_range(n_tiles_c_out):
#             for input_row in nl.affine_range(out_pool_height//2):
#                     # create the accumulator in psum
#                     X_out_tile_before_pooling = nl.zeros((c_in_pmax, 2*out_width),  X_out.dtype, buffer=nl.psum)
#                     for input_tile_idx in nl.affine_range(n_tiles_c_in):
#                         for i in nl.affine_range(filter_height):
#                             for j in nl.affine_range(filter_width):
#                                         # Declare the tiles on SBUF
#                                 W_tile = nl.ndarray((TILE_128, TILE_128, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
#                                 X_rhs_tile = nl.ndarray((TILE_128, TILE_WIDTH), dtype=X.dtype, buffer=nl.sbuf)
#                                 X_rhs_tile_shifted = nl.ndarray((TILE_128, 2*out_width), dtype=X.dtype, buffer=nl.sbuf)
#                                 # shift row
#                                 shifted_row = 2*input_row*pool_size+i
#                                 X_rhs_tile[...] = nl.load(X[b,c_in_pmax*input_tile_idx:c_in_pmax*(input_tile_idx+1), shifted_row:shifted_row+2, :]).reshape((TILE_128, TILE_WIDTH))
#                                 pdb.set_trace()
#                                 # shift the columns
#                                 X_rhs_tile_shifted[...] = nl.copy(X_rhs_tile[in_channels_idx, nl.arange(2)[None,:,None]*input_width+(out_width_idx+j)]).reshape((TILE_128, 2*out_width))
#                                 W_tile[...] = nl.load(W[c_in_pmax*output_tile_idx:c_in_pmax*(output_tile_idx+1), c_in_pmax*input_tile_idx:c_in_pmax*(input_tile_idx+1)])
                                
#                                 # do matmul with W
#                                 X_out_tile_before_pooling += nl.matmul(W_tile[:,:,i,j], X_rhs_tile_shifted, transpose_x=False)
#                     # add bias
#                     X_out_tile_before_pooling_with_bias = (nl.load(bias[output_tile_idx*c_in_pmax:(output_tile_idx+1)*(c_in_pmax),])+nl.copy(X_out_tile_before_pooling))
                    
#                     # perform maxpooling
#                     temp1 = nl.copy(X_out_tile_before_pooling_with_bias).reshape((c_in_pmax, pool_size, 2/pool_size, out_width))
#                     temp2 = nl.max(temp1, axis=1)
#                     temp3 = nl.copy(temp2).reshape((c_in_pmax, 2/pool_size, pool_size, out_width/pool_size))
#                     X_out_tile_after_pooling = nl.max(temp3, axis=2)
#                     # X_out_tile_after_pooling = nl.max(nl.max(X_out_tile_before_pooling_with_bias.reshape((c_in_pmax, pool_size, 2/pool_size, out_width)), axis=1).reshape((c_in_pmax, 2/pool_size, pool_size, out_width/pool_size)), axis=2)
#                     nl.store(X_out[b, c_in_pmax*output_tile_idx:c_in_pmax*(output_tile_idx+1), 2*input_row:(2*input_row+2)], value=X_out_tile_after_pooling)
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
                
                partial_sum = nl.zeros((TILE_128, 2*out_width), X_out.dtype, buffer=nl.psum)
                for input_tile_idx in nl.affine_range(n_tiles_c_in):
                    W_tile = nl.ndarray((TILE_128, TILE_128, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            X_tile = nl.ndarray((TILE_128, 2, input_width), dtype=X.dtype, buffer=nl.sbuf)
                            X_tile[...] = nl.load(X[b, input_tile_idx*c_in_pmax:(input_tile_idx+1)*c_in_pmax, i+output_row*2:i+output_row*2+2,:])
                            X_tile_shifted = nl.copy(X_tile)[in_channels_idx, nl.arange(2)[None,:,None], out_width_idx+j]
                            X_tile_reshaped = nl.copy(X_tile_shifted).reshape((TILE_128, 2*out_width))
                            W_tile[...] = nl.load(W[output_tile_idx*c_in_pmax:(output_tile_idx+1)*c_in_pmax, input_tile_idx*c_in_pmax:(input_tile_idx+1)*c_in_pmax])
                            partial_sum += nl.matmul(nl.copy(W_tile[:,:,i,j]), X_tile_reshaped, transpose_x=False)
                temp = nl.copy(partial_sum).reshape((c_in_pmax, 2, out_width))
                nl.store(X_out[b, output_tile_idx*c_in_pmax:(output_tile_idx+1)*c_in_pmax, 2*output_row:2*output_row+2], value=temp[...])
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