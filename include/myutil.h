#include <cuda.h>

inline __host__ __device__ unsigned int iDivUp(unsigned int a, unsigned int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline __host__ __device__ unsigned int iAlignUp(unsigned int a, unsigned int b)
{
    return (a % b != 0) ?  (a - a % b + b) : a;
}

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{   
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}


inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}

inline __host__ __device__ float2 conj(float2 v)
{
    return make_float2(v.x, -v.y);
}
inline __host__ __device__ float2 complex_mul(float2 a, float2 b)
{
    return make_float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
inline __host__ __device__ float2 conj_mul(float2 a, float2 b)
{
    return make_float2(a.x*b.x+a.y*b.y, a.y*b.x-a.x*b.y);
}

