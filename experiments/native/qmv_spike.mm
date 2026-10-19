// Native Metal spike: how close to the memory roof can a 4-bit matvec get on M5?
//
//   clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation \
//       experiments/native/qmv_spike.mm -o experiments/native/qmv_spike && experiments/native/qmv_spike
//
// Layout = MLX affine 4-bit, group 64: w uint32 [N, K/8], scales/biases half [N, K/64].

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char* kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

// ---- pure read bandwidth roof
kernel void read_roof(device const uint4* buf [[buffer(0)]],
                      device uint* out [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint gsize [[threads_per_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint4 acc = 0;
    for (uint i = gid; i < n; i += gsize) acc += buf[i];
    uint v = simd_sum(acc.x + acc.y + acc.z + acc.w);
    if (lane == 0) out[gid / 32] = v;
}

// ---- 4-bit matvec: ROWS rows per simdgroup, PACKS uint32 (8 weights each) per lane per step
template <int ROWS, int PACKS>
kernel void qmv(device const uint* w [[buffer(0)]],
                device const half* scales [[buffer(1)]],
                device const half* biases [[buffer(2)]],
                device const half* x [[buffer(3)]],
                device float* y [[buffer(4)]],
                constant uint& N [[buffer(5)]],
                constant uint& K [[buffer(6)]],
                uint lane [[thread_index_in_simdgroup]],
                uint sg [[simdgroup_index_in_threadgroup]],
                uint sgs [[simdgroups_per_threadgroup]],
                uint tg [[threadgroup_position_in_grid]]) {
    constexpr int VPT = PACKS * 8;
    const uint row0 = (tg * sgs + sg) * ROWS;
    if (row0 >= N) return;
    const uint KW = K / 8, G = K / 64, STEP = 32 * VPT;

    float acc[ROWS];
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (uint k0 = lane * VPT; k0 < K; k0 += STEP) {
        float xv[VPT];
        float xs = 0.0f;
        for (int i = 0; i < VPT; ++i) { xv[i] = float(x[k0 + i]); xs += xv[i]; }
        const uint g = k0 / 64;
        for (int r = 0; r < ROWS; ++r) {
            const uint row = row0 + r;
            device const uint* wp = w + row * KW + k0 / 8;
            uint words[PACKS];
            for (int p = 0; p < PACKS; ++p) words[p] = wp[p];
            float d = 0.0f;
            for (int p = 0; p < PACKS; ++p)
                for (int j = 0; j < 8; ++j)
                    d += float((words[p] >> (4 * j)) & 0xFu) * xv[p * 8 + j];
            acc[r] += float(scales[row * G + g]) * d + float(biases[row * G + g]) * xs;
        }
    }
    for (int r = 0; r < ROWS; ++r) {
        const float v = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < N) y[row0 + r] = v;
    }
}

#define INST(R, P) \
  template [[host_name("qmv_r" #R "_p" #P)]] kernel void qmv<R, P>( \
      device const uint*, device const half*, device const half*, device const half*, \
      device float*, constant uint&, constant uint&, uint, uint, uint, uint);
INST(2, 2) INST(4, 2) INST(8, 2) INST(2, 4) INST(4, 4) INST(8, 4) INST(4, 8)
)METAL";

static uint32_t rng_state = 12345;
static inline uint32_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }

static uint16_t f2h(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float h2f(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (float)h; }

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", dev.name.UTF8String);
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeFast;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSource) options:opts error:&err];
        if (!lib) { printf("compile error: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        const uint32_t N = 131072, K = 4096;  // 256 MB packed + 32 MB scales/biases: far beyond the SLC
        const size_t wBytes = (size_t)N * K / 2, sbCount = (size_t)N * K / 64;
        id<MTLBuffer> W = [dev newBufferWithLength:wBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> S = [dev newBufferWithLength:sbCount * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> B = [dev newBufferWithLength:sbCount * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> X = [dev newBufferWithLength:K * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
        uint32_t* w = (uint32_t*)W.contents;
        for (size_t i = 0; i < wBytes / 4; ++i) w[i] = rnd();
        uint16_t *s = (uint16_t*)S.contents, *b = (uint16_t*)B.contents, *x = (uint16_t*)X.contents;
        for (size_t i = 0; i < sbCount; ++i) { s[i] = f2h(0.001f + (rnd() % 1000) * 1e-5f); b[i] = f2h(-0.008f); }
        for (uint32_t i = 0; i < K; ++i) x[i] = f2h(((int)(rnd() % 2001) - 1000) * 1e-3f);

        // CPU reference for the first rows
        const int CHECK = 64;
        std::vector<double> ref(CHECK);
        for (int r = 0; r < CHECK; ++r) {
            double acc = 0;
            for (uint32_t k = 0; k < K; ++k) {
                uint32_t q4 = (w[(size_t)r * K / 8 + k / 8] >> (4 * (k % 8))) & 0xF;
                acc += (h2f(s[(size_t)r * K / 64 + k / 64]) * q4 + h2f(b[(size_t)r * K / 64 + k / 64])) * h2f(x[k]);
            }
            ref[r] = acc;
        }

        const int ITERS = 60;
        auto time_dispatches = [&](id<MTLComputePipelineState> pso, void (^encode)(id<MTLComputeCommandEncoder>)) {
            for (int warm = 0; warm < 2; ++warm) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                encode(enc);
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            for (int i = 0; i < ITERS; ++i) encode(enc);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) / ITERS;
        };

        // ---- roof
        {
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"read_roof"] error:&err];
            uint32_t n = (uint32_t)(wBytes / 16);
            id<MTLBuffer> out = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
            for (uint32_t threads : {8192u, 32768u, 131072u}) {
                double t = time_dispatches(pso, ^(id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:W offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1];
                    [e setBytes:&n length:4 atIndex:2];
                    [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                });
                printf("read roof (%6u threads): %6.1f GB/s\n", threads, wBytes / t / 1e9);
            }
        }

        // ---- qmv variants
        const double qBytes = wBytes + sbCount * 4.0;
        struct V { const char* name; int rows; };
        V variants[] = {{"qmv_r2_p2", 2}, {"qmv_r4_p2", 4}, {"qmv_r8_p2", 8}, {"qmv_r2_p4", 2},
                        {"qmv_r4_p4", 4}, {"qmv_r8_p4", 8}, {"qmv_r4_p8", 4}};
        for (auto& v : variants) {
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(v.name)] error:&err];
            if (!pso) { printf("%s: %s\n", v.name, err.localizedDescription.UTF8String); continue; }
            for (int sgs : {2, 4, 8}) {
                const uint32_t simdgroups = N / v.rows;
                const uint32_t tgs = (simdgroups + sgs - 1) / sgs;
                memset(Y.contents, 0, N * 4);
                double t = time_dispatches(pso, ^(id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:W offset:0 atIndex:0]; [e setBuffer:S offset:0 atIndex:1];
                    [e setBuffer:B offset:0 atIndex:2]; [e setBuffer:X offset:0 atIndex:3];
                    [e setBuffer:Y offset:0 atIndex:4];
                    [e setBytes:&N length:4 atIndex:5]; [e setBytes:&K length:4 atIndex:6];
                    [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
                });
                const float* y = (const float*)Y.contents;
                double maxrel = 0;
                for (int r = 0; r < CHECK; ++r) maxrel = fmax(maxrel, fabs(y[r] - ref[r]) / (fabs(ref[r]) + 1e-3));
                printf("%-10s sg/tg=%d: %7.3f ms  %6.1f GB/s  max rel.err %.1e\n", v.name, sgs, t * 1e3, qBytes / t / 1e9, maxrel);
            }
        }
    }
    return 0;
}
