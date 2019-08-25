/*
Author: Cao Thanh Tung and Zheng Jiaqi
Date: 21/01/2010, 20/08/2019

File Name: pba2DKernel.h

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

#define TOID(x, y, size)    (__mul24((y), (size)) + (x))

__global__ void kernelFloodDown(short2 *input, short2 *output, int size, int bandSize) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * bandSize; 
    int id = TOID(tx, ty, size); 

    short2 pixel1, pixel2; 

    pixel1 = make_short2(MARKER, MARKER); 

    for (int i = 0; i < bandSize; i++, id += size) {
        pixel2 = input[id]; 

        if (pixel2.x != MARKER) 
            pixel1 = pixel2; 

        output[id] = pixel1; 
    }
}

__global__ void kernelFloodUp(short2 *input, short2 *output, int size, int bandSize) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = (blockIdx.y+1) * bandSize - 1; 
    int id = TOID(tx, ty, size); 

    short2 pixel1, pixel2; 
    int dist1, dist2; 

    pixel1 = make_short2(MARKER, MARKER); 

    for (int i = 0; i < bandSize; i++, id -= size) {
        dist1 = abs(pixel1.y - ty + i); 

        pixel2 = input[id]; 
        dist2 = abs(pixel2.y - ty + i); 

        if (dist2 < dist1) 
            pixel1 = pixel2; 

        output[id] = pixel1; 
    }
}

__global__ void kernelPropagateInterband(short2 *input, short2 *output, int size, int bandSize) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int inc = bandSize * size; 
    int ny, nid, nDist; 
    short2 pixel; 

    // Top row, look backward
    int ty = blockIdx.y * bandSize; 
    int topId = TOID(tx, ty, size); 
    int bottomId = TOID(tx, ty + bandSize - 1, size); 
    int tid = blockIdx.y * size + tx;
    int bid = tid + (size * size / bandSize);

    pixel = input[topId]; 
    int myDist = abs(pixel.y - ty); 
    output[tid] = pixel;

    for (nid = bottomId - inc; nid >= 0; nid -= inc) {
        pixel = input[nid]; 

        if (pixel.x != MARKER) { 
            nDist = abs(pixel.y - ty); 

            if (nDist < myDist)
                output[tid] = pixel;

            break;  
        }
    }

    // Last row, look downward
    ty = ty + bandSize - 1; 
    pixel = input[bottomId]; 
    myDist = abs(pixel.y - ty); 
    output[bid] = pixel;

    for (ny = ty + 1, nid = topId + inc; ny < size; ny += bandSize, nid += inc) {
        pixel = input[nid]; 

        if (pixel.x != MARKER) { 
            nDist = abs(pixel.y - ty); 

            if (nDist < myDist)
                output[bid] = pixel;

            break; 
        }
    }
}

__global__ void kernelUpdateVertical(short2 *color, short2 *margin, short2 *output, int size, int bandSize) 
{
    __shared__ short2 block[BLOCKSIZE][BLOCKSIZE];

    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * bandSize; 

    short2 top = margin[blockIdx.y * size + tx]; 
    short2 bottom = margin[(blockIdx.y + size / bandSize) * size + tx]; 
    short2 pixel; 

    int dist, myDist; 

    int id = TOID(tx, ty, size); 

    int n_step = bandSize / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        int y_start = blockIdx.y * bandSize + step * blockDim.x;
        int y_end = y_start + blockDim.x;

        for (ty = y_start; ty < y_end; ++ty, id += size) {
            pixel = color[id]; 
            myDist = abs(pixel.y - ty); 

            dist = abs(top.y - ty);
            if (dist < myDist) { myDist = dist; pixel = top; }

            dist = abs(bottom.y - ty); 
            if (dist < myDist) pixel = bottom; 

            block[threadIdx.x][ty - y_start] = make_short2(pixel.y, pixel.x);
        }

        __syncthreads();

        int tid = TOID(blockIdx.y * bandSize + step * blockDim.x + threadIdx.x, \
                        blockIdx.x * blockDim.x, size);

        for(int i = 0; i < blockDim.x; ++i, tid += size) {
            output[tid] = block[i][threadIdx.x];
        }

        __syncthreads();
    }
}

#define LL long long
__device__ bool dominate(LL x1, LL y1, LL x2, LL y2, LL x3, LL y3, LL x0)
{
    LL k1 = y2 - y1, k2 = y3 - y2;
    return (k1 * (y1 + y2) + (x2 - x1) * ((x1 + x2) - (x0 << 1))) * k2 > \
            (k2 * (y2 + y3) + (x3 - x2) * ((x2 + x3) - (x0 << 1))) * k1;
}
#undef LL

__global__ void kernelProximatePoints(short2 *input, short2 *stack, int size, int bandSize) 
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
    int ty = __mul24(blockIdx.y, bandSize); 
    int id = TOID(tx, ty, size); 
    int lasty = -1; 
    short2 last1, last2, current; 

    last1.y = -1; last2.y = -1; 

    for (int i = 0; i < bandSize; i++, id += size) {
        current = input[id];

        if (current.x != MARKER) {
            while (last2.y >= 0) {
                if (!dominate(last1.x, last2.y, last2.x, \
                    lasty, current.x, current.y, tx))
                    break;

                lasty = last2.y; last2 = last1; 

                if (last1.y >= 0)
                    last1 = stack[TOID(tx, last1.y, size)]; 
            }

            last1 = last2; last2 = make_short2(current.x, lasty); lasty = current.y; 

            stack[id] = last2;
        }
    }

    // Store the pointer to the tail at the last pixel of this band
    if (lasty != ty + bandSize - 1) 
        stack[TOID(tx, ty + bandSize - 1, size)] = make_short2(MARKER, lasty); 
}

__global__ void kernelCreateForwardPointers(short2 *input, short2 *output, int size, int bandSize) 
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
    int ty = __mul24(blockIdx.y+1, bandSize) - 1; 
    int id = TOID(tx, ty, size); 
    int lasty = -1, nexty; 
    short2 current; 

    // Get the tail pointer
    current = input[id]; 

    if (current.x == MARKER)
        nexty = current.y; 
    else
        nexty = ty; 

    for (int i = 0; i < bandSize; i++, id -= size)
        if (ty - i == nexty) {
            current = make_short2(lasty, input[id].y);
            output[id] = current; 

            lasty = nexty; 
            nexty = current.y; 
        }

    // Store the pointer to the head at the first pixel of this band
    if (lasty != ty - bandSize + 1) 
        output[id + size] = make_short2(lasty, MARKER);  
}

__global__ void kernelMergeBands(short2 *color, short2 *link, short2 *output, int size, int bandSize)
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
    int band1 = blockIdx.y * 2; 
    int band2 = band1 + 1; 
    int firsty, lasty; 
    short2 last1, last2, current; 
    // last1 and last2: x component store the x coordinate of the site, 
    // y component store the backward pointer
    // current: y component store the x coordinate of the site, 
    // x component store the forward pointer

    // Get the two last items of the first list
    lasty = __mul24(band2, bandSize) - 1; 
    last2 = make_short2(color[TOID(tx, lasty, size)].x, 
        link[TOID(tx, lasty, size)].y); 

    if (last2.x == MARKER) {
        lasty = last2.y; 

        if (lasty >= 0) 
            last2 = make_short2(color[TOID(tx, lasty, size)].x, 
            link[TOID(tx, lasty, size)].y); 
        else
            last2 = make_short2(MARKER, MARKER); 
    }

    if (last2.y >= 0) {
        // Second item at the top of the stack
        last1 = make_short2(color[TOID(tx, last2.y, size)].x, 
            link[TOID(tx, last2.y, size)].y); 
    }

    // Get the first item of the second band
    firsty = __mul24(band2, bandSize); 
    current = make_short2(link[TOID(tx, firsty, size)].x, 
        color[TOID(tx, firsty, size)].x); 

    if (current.y == MARKER) {
        firsty = current.x; 

        if (firsty >= 0) 
            current = make_short2(link[TOID(tx, firsty, size)].x, 
            color[TOID(tx, firsty, size)].x); 
        else
            current = make_short2(MARKER, MARKER); 
    }

    // Count the number of item in the second band that survive so far. 
    // Once it reaches 2, we can stop. 
    int top = 0; 

    while (top < 2 && current.y >= 0) {
        // While there's still something on the left
        while (last2.y >= 0) {

            if (!dominate(last1.x, last2.y, last2.x, \
                lasty, current.y, firsty, tx)) 
                break; 

            lasty = last2.y; last2 = last1; 
            top--; 

            if (last1.y >= 0) 
                last1 = make_short2(color[TOID(tx, last1.y, size)].x, 
                link[TOID(tx, last1.y, size)].y); 
        }

        // Update the current pointer 
        output[TOID(tx, firsty, size)] = make_short2(current.x, lasty); 

        if (lasty >= 0) 
            output[TOID(tx, lasty, size)] = make_short2(firsty, last2.y); 

        last1 = last2; last2 = make_short2(current.y, lasty); lasty = firsty; 
        firsty = current.x; 

        top = max(1, top + 1); 

        // Advance the current pointer to the next one
        if (firsty >= 0) 
            current = make_short2(link[TOID(tx, firsty, size)].x, 
            color[TOID(tx, firsty, size)].x); 
        else
            current = make_short2(MARKER, MARKER); 
    }

    // Update the head and tail pointer. 
    firsty = __mul24(band1, bandSize); 
    lasty = __mul24(band2, bandSize); 
    current = link[TOID(tx, firsty, size)]; 

    if (current.y == MARKER && current.x < 0) {	// No head?
        last1 = link[TOID(tx, lasty, size)]; 

        if (last1.y == MARKER)
            current.x = last1.x; 
        else
            current.x = lasty; 

        output[TOID(tx, firsty, size)] = current; 
    }

    firsty = __mul24(band1, bandSize) + bandSize - 1; 
    lasty = __mul24(band2, bandSize) + bandSize - 1; 
    current = link[TOID(tx, lasty, size)]; 

    if (current.x == MARKER && current.y < 0) {	// No tail?
        last1 = link[TOID(tx, firsty, size)]; 

        if (last1.x == MARKER) 
            current.y = last1.y; 
        else
            current.y = firsty; 

        output[TOID(tx, lasty, size)] = current; 
    }
}

__global__ void kernelDoubleToSingleList(short2 *color, short2 *link, short2 *output, int size)
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
    int ty = blockIdx.y; 
    int id = TOID(tx, ty, size); 

    output[id] = make_short2(color[id].x, link[id].y); 
}

__global__ void kernelColor(short2 *input, short2 *output, int size) 
{
    __shared__ short2 block[BLOCKSIZE][BLOCKSIZE];

    int col = threadIdx.x; 
    int tid = threadIdx.y; 
    int tx = __mul24(blockIdx.x, blockDim.x) + col; 
    int dx, dy, lasty; 
    unsigned int best, dist; 
    short2 last1, last2; 

    lasty = size - 1; 

    last2 = input[TOID(tx, lasty, size)];

    if (last2.x == MARKER) {
        lasty = last2.y; 
        last2 = input[TOID(tx, lasty, size)];
    }

    if (last2.y >= 0) 
        last1 = input[TOID(tx, last2.y, size)];

    int y_start, y_end, n_step = size / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        y_start = size - step * blockDim.x - 1;
        y_end = size - (step + 1) * blockDim.x;

        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            dx = last2.x - tx; dy = lasty - ty; 
            best = dist = __mul24(dx, dx) + __mul24(dy, dy); 

            while (last2.y >= 0) {
                dx = last1.x - tx; dy = last2.y - ty; 
                dist = __mul24(dx, dx) + __mul24(dy, dy); 

                if (dist > best) 
                    break; 

                best = dist; lasty = last2.y; last2 = last1;

                if (last2.y >= 0) 
                    last1 = input[TOID(tx, last2.y, size)];
            }

            block[threadIdx.x][ty - y_end] = make_short2(lasty, last2.x);
        }

        __syncthreads();

        if(!threadIdx.y) {
            int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x, size);
            for(int i = 0; i < blockDim.x; ++i, id+=size) {
                output[id] = block[i][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
}
