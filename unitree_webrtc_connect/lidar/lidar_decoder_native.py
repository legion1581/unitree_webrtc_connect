# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import lz4.block

def decompress(compressed_data, decomp_size):
    decompressed = lz4.block.decompress(
        compressed_data,
        uncompressed_size=decomp_size
    )
    return decompressed

def bits_to_points(buf, origin, resolution=0.05):
    buf = np.frombuffer(bytearray(buf), dtype=np.uint8)
    nonzero_indices = np.nonzero(buf)[0]

    if len(nonzero_indices) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Get byte values and unpack to bits (MSB first matches original logic)
    byte_values = buf[nonzero_indices]
    bits = np.unpackbits(byte_values).reshape(-1, 8)

    # Calculate base coordinates for each nonzero byte
    z = nonzero_indices // 0x800
    n_slice = nonzero_indices % 0x800
    y = n_slice // 0x10
    x_base = (n_slice % 0x10) * 8

    # Expand coordinates to match 8 bits per byte
    z_expanded = np.repeat(z, 8)
    y_expanded = np.repeat(y, 8)
    x = np.repeat(x_base, 8) + np.tile(np.arange(8), len(nonzero_indices))

    # Filter to only points where bit is set
    mask = bits.ravel() == 1
    points = np.column_stack((x[mask], y_expanded[mask], z_expanded[mask]))

    return points * resolution + origin

class LidarDecoder:
    def decode(self, compressed_data, data):
        decompressed = decompress(compressed_data, data["src_size"])
        points = bits_to_points(decompressed, data["origin"], data["resolution"])

        return {
                "points": points,
                # "raw": compressed_data,
        }
