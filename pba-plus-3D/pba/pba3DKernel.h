/*
Author: Cao Thanh Tung, Zheng Jiaqi
Date: 21/01/2010, 25/08/2019

File Name: pba3DKernel.h

===============================================================================

Copyright (c) 2019, School of Computing, National University of Singapore. 
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/pba.html

If you use PBA and you like it or have comments on its usefulness etc., we 
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#define TOID(x, y, z, size)    ((((z) * (size)) + (y)) * (size) + (x))

// Flood along the Z axis
__global__ void kernelFloodZ(int *input, int *output, int size) 
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
	int ty = blockIdx.y * blockDim.y + threadIdx.y; 
	int tz = 0; 

    int plane = size * size; 
    int id = TOID(tx, ty, tz, size); 
    int pixel1, pixel2; 

    pixel1 = ENCODE(0,0,0,1,0); 

    // Sweep down
    for (int i = 0; i < size; i++, id += plane) {
        pixel2 = input[id];

        if (!NOTSITE(pixel2))
            pixel1 = pixel2;

        output[id] = pixel1;
    }

	int dist1, dist2, nz;

	id -= plane + plane;

    // Sweep up
    for (int i = size - 2; i >= 0; i--, id -= plane) {
        nz = GET_Z(pixel1);
        dist1 = abs(nz - (tz + i));

        pixel2 = output[id];
        nz = GET_Z(pixel2);
        dist2 = abs(nz - (tz + i));

        if (dist2 < dist1)
            pixel1 = pixel2;

        output[id] = pixel1;
    }
}

#define LL long long
__device__ bool dominate(LL x_1, LL y_1, LL z_1, LL x_2, LL y_2, LL z_2, LL x_3, LL y_3, LL z_3, LL x_0, LL z_0)
{
	LL k_1 = y_2 - y_1, k_2 = y_3 - y_2;

	return (((y_1 + y_2) * k_1 + ((x_2 - x_1) * (x_1 + x_2 - (x_0 << 1)) + (z_2 - z_1) * (z_1 + z_2 - (z_0 << 1)))) * k_2 > \
			((y_2 + y_3) * k_2 + ((x_3 - x_2) * (x_2 + x_3 - (x_0 << 1)) + (z_3 - z_2) * (z_2 + z_3 - (z_0 << 1)))) * k_1);
}
#undef LL

__global__ void kernelMaurerAxis(int *input, int *stack, int size) 
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int tz = blockIdx.y * blockDim.y + threadIdx.y;
	int ty = 0;

	int id = TOID(tx, ty, tz, size);

    int lasty = 0;
    int x1, y1, z1, x2, y2, z2, nx, ny, nz;
    int p = ENCODE(0,0,0,1,0), s1 = ENCODE(0,0,0,1,0), s2 = ENCODE(0,0,0,1,0);
    int flag = 0;

    for (ty = 0; ty < size; ++ty, id += size) {
        p = input[id];

        if (!NOTSITE(p)) {

            while (HASNEXT(s2)) {
                DECODE(s1, x1, y1, z1);
                DECODE(s2, x2, y2, z2);
                DECODE(p, nx, ny, nz);

                if (!dominate(x1, y2, z1, x2, lasty, z2, nx, ty, nz, tx, tz))
                	break;

                lasty = y2; s2 = s1; y2 = y1;

                if (HASNEXT(s2))
                    s1 = stack[TOID(tx, y2, tz, size)];
            }

            DECODE(p, nx, ny, nz);
            s1 = s2;
            s2 = ENCODE(nx, lasty, nz, 0, flag);
            y2 = lasty;
            lasty = ty;

            stack[id] = s2;

            flag = 1;
        }
    }

    if (NOTSITE(p))
        stack[TOID(tx, ty - 1, tz, size)] = ENCODE(0, lasty, 0, 1, flag); 
}

__global__ void kernelColorAxis(int *input, int *output, int size) 
{
	__shared__ int block[BLOCKSIZE][BLOCKSIZE];

	int col = threadIdx.x;
	int tid = threadIdx.y;
	int tx = blockIdx.x * blockDim.x + col; 
	int tz = blockIdx.y;
 
    int x1, y1, z1, x2, y2, z2;
    int last1 = ENCODE(0,0,0,1,0), last2 = ENCODE(0,0,0,1,0), lasty;
    long long dx, dy, dz, best, dist;

	lasty = size - 1;

	last2 = input[TOID(tx, lasty, tz, size)]; 
	DECODE(last2, x2, y2, z2);

	if (NOTSITE(last2)) {
		lasty = y2;
		if(HASNEXT(last2)) {
			last2 = input[TOID(tx, lasty, tz, size)];
			DECODE(last2, x2, y2, z2);
		}
	}

    if (HASNEXT(last2)) {
		last1 = input[TOID(tx, y2, tz, size)];
		DECODE(last1, x1, y1, z1);
	}

	int y_start, y_end, n_step = size / blockDim.x;
	for(int step = 0; step < n_step; ++step) {
		y_start = size - step * blockDim.x - 1;
		y_end = size - (step + 1) * blockDim.x;

	    for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
	    	dx = x2 - tx; dy = lasty - ty; dz = z2 - tz;
			best = dx * dx + dy * dy + dz * dz;

			while (HASNEXT(last2)) {
				dx = x1 - tx; dy = y2 - ty; dz = z1 - tz;
				dist = dx * dx + dy * dy + dz * dz;

				if(dist > best) break;

				best = dist; lasty = y2; last2 = last1;
				DECODE(last2, x2, y2, z2);

				if (HASNEXT(last2)) {
					last1 = input[TOID(tx, y2, tz, size)];
					DECODE(last1, x1, y1, z1); 
				}
	        }

	        block[threadIdx.x][ty - y_end] = ENCODE(lasty, x2, z2, NOTSITE(last2), 0);
	    }

	    __syncthreads();

	    if(!threadIdx.y) {
	    	int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x, tz, size);
	    	for(int i = 0; i < blockDim.x; i++, id+=size) {
	    		output[id] = block[i][threadIdx.x];
	    	}
	    }

	    __syncthreads();
	}
}
