/*
Author: Cao Thanh Tung, Zheng Jiaqi
Date: 21/01/2010, 25/08/2019

File Name: pba3D.h

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

#ifndef __CUDA_H__
#define __CUDA_H__

// Initialize CUDA and allocate memory
// textureSize is 2^k with k >= 5
extern "C" void pba3DInitialization(int textureSize); 

// Deallocate memory in GPU
extern "C" void pba3DDeinitialization(); 

// Compute 3D Voronoi diagram
// Input: a 3D texture. Each pixel is an integer encoding 3 coordinates. 
// 		For each site at (x, y, z), the pixel at coordinate (x, y, z) should contain 
// 		the encoded coordinate (x, y, z). Pixels that are not sites should contain 
// 		the integer MARKER. Use ENCODE (and DECODE) macro to encode (and decode).
// Output: a 3D texture. Each pixel is an integer encoding 3 coordinates 
// 		of its nearest site. 
// See our website for the effect of the three parameters: 
// 		phase1Band, phase2Band, phase3Band
// Parameters must divide textureSize
extern "C" void pba3DVoronoiDiagram(int *input, int *output, 
                                    int phase1Band, int phase2Band, int phase3Band);

#define MARKER	    -2147483648
#define MAX_INT 	2147483647

// Sites 	 : ENCODE(x, y, z, 0, 0)
// Not sites : ENCODE(0, 0, 0, 1, 0) or MARKER
#define ENCODE(x, y, z, a, b)  (((x) << 20) | ((y) << 10) | (z) | ((a) << 31) | ((b) << 30))
#define DECODE(value, x, y, z) \
    x = ((value) >> 20) & 0x3ff; \
    y = ((value) >> 10) & 0x3ff; \
    z = (value) & 0x3ff

#define NOTSITE(value)	(((value) >> 31) & 1)
#define HASNEXT(value) 	(((value) >> 30) & 1)

#define GET_X(value)	(((value) >> 20) & 0x3ff)
#define GET_Y(value)	(((value) >> 10) & 0x3ff)
#define GET_Z(value)	((NOTSITE((value))) ? MAX_INT : ((value) & 0x3ff))

#endif
