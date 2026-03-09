// Radiance cascade draw: SDF brush + color pick, then seed/JFA/distance/RC/overlay
// so drawn strokes act as light sources. Requires maxStorageTexturesPerShaderStage >= 8
// (requested in App.vue). Uses 7 storage textures: scene×2, seedPacked, jfaPacked,
// distanceTex, radianceTex, outputTexture.

import playground;
import rendering;

static const uint THREADS = 16;
static const float PI = 3.14159265;
static const float TAU = 2.0 * PI;
static const uint NO_SEED_PACKED = 0xFFFFFFFFu;

uint packRGBA(float r, float g, float b, float a)
{
    uint ur = uint(clamp(r, 0.0, 1.0) * 255.0 + 0.5);
    uint ug = uint(clamp(g, 0.0, 1.0) * 255.0 + 0.5);
    uint ub = uint(clamp(b, 0.0, 1.0) * 255.0 + 0.5);
    uint ua = uint(clamp(a, 0.0, 1.0) * 255.0 + 0.5);
    return ur | (ug << 8) | (ub << 16) | (ua << 24);
}

float4 unpackRGBA(uint packed)
{
    float r = float(packed & 255u) / 255.0;
    float g = float((packed >> 8) & 255u) / 255.0;
    float b = float((packed >> 16) & 255u) / 255.0;
    float a = float((packed >> 24) & 255u) / 255.0;
    return float4(r, g, b, a);
}

uint packSeed(float u, float v)
{
    uint u16 = uint(clamp(u, 0.0, 1.0) * 65535.0 + 0.5);
    uint v16 = uint(clamp(v, 0.0, 1.0) * 65535.0 + 0.5);
    return u16 | (v16 << 16);
}

float2 unpackSeed(uint packed)
{
    if (packed == NO_SEED_PACKED) return float2(2.0, 2.0);
    float u = float(packed & 65535u) / 65535.0;
    float v = float(packed >> 16) / 65535.0;
    return float2(u, v);
}

[format("r32ui")]
[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<uint> sceneRead;
[format("r32ui")]
[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<uint> sceneWrite;

[format("r32ui")]
[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<uint> seedPacked;
[format("r32ui")]
[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<uint> jfaPacked;

[format("r32f")]
[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<float> distanceTex;

[format("r32f")]
[playground::BLACK_SCREEN(0.25, 0.25)]
RWTexture2D<float> radianceTex;

[playground::ZEROS(4)]
RWStructuredBuffer<float> lastMouse;

[playground::SLIDER(6.0, 2.0, 32.0)]
uniform float brushRadius;
[playground::COLOR_PICK(1.0, 0.9, 0.6)]
uniform float3 colorRGB;
[playground::MOUSE_POSITION]
uniform float4 mousePosition;

[playground::SLIDER(4.0, 2.0, 16.0)]
uniform float basePixelsBetweenProbes;

[playground::SLIDER(8.0, 4.0, 32.0)]
uniform float rayCount;

float sdfLineSquared(float2 p, float2 from, float2 to)
{
    float2 toStart = p - from;
    float2 line = to - from;
    float lineLengthSquared = dot(line, line);
    float t = clamp(dot(toStart, line) / max(lineLengthSquared, 1e-6), 0.0, 1.0);
    float2 closest = toStart - line * t;
    return dot(closest, closest);
}

[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("sceneWrite")]
void copy(uint2 tid: SV_DispatchThreadID)
{
    int2 c = int2(tid);
    sceneRead[c] = sceneWrite.Load(c);
}

[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("sceneWrite")]
void draw(uint2 tid: SV_DispatchThreadID)
{
    int2 coord = int2(tid);
    float2 coordF = float2(tid) + 0.5;
    uint2 dims;
    sceneWrite.GetDimensions(dims.x, dims.y);
    float height = float(dims.y);
    float fromX = lastMouse[0];
    float fromY = lastMouse[1];
    float2 to = float2(mousePosition.x, height - mousePosition.y);
    bool drawing = mousePosition.z < 0.0;
    bool hasFrom = (fromX != 0.0 || fromY != 0.0);
    float2 from = (drawing && !hasFrom) ? to : float2(fromX, height - fromY);
    float radiusSq = brushRadius * brushRadius;
    float4 current = unpackRGBA(sceneRead.Load(coord));
    float r = current.r, g = current.g, b = current.b, a = current.a;
    if (drawing)
    {
        float distSq = sdfLineSquared(coordF, from, to);
        if (distSq <= radiusSq)
        {
            r = colorRGB.r; g = colorRGB.g; b = colorRGB.b; a = 1.0;
        }
    }
    sceneWrite[coord] = packRGBA(r, g, b, a);
}

[shader("compute")]
[numthreads(1, 1, 1)]
[playground::CALL(1, 1, 1)]
void updateLastMouse(uint2 _: SV_DispatchThreadID)
{
    bool drawing = mousePosition.z < 0.0;
    lastMouse[0] = drawing ? mousePosition.x : 0.0;
    lastMouse[1] = drawing ? mousePosition.y : 0.0;
}

[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("sceneWrite")]
void seed(uint2 tid: SV_DispatchThreadID)
{
    uint2 dims;
    sceneWrite.GetDimensions(dims.x, dims.y);
    float2 uv = (float2(tid) + 0.5) / float2(dims);
    float4 c = unpackRGBA(sceneWrite.Load(int2(tid)));
    seedPacked[tid] = (c.a > 0.01) ? packSeed(uv.x, uv.y) : NO_SEED_PACKED;
}

void jfaStepSeedToJfa(uint2 tid, uint2 dims, uint offset)
{
    float2 uv = (float2(tid) + 0.5) / float2(dims);
    float2 nearestSeed = float2(2.0, 2.0);
    float nearestDist = 1e6;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0) continue;
            int2 s = int2(tid) + int2(dx, dy) * int(offset);
            if (s.x < 0 || s.x >= int(dims.x) || s.y < 0 || s.y >= int(dims.y)) continue;
            float2 seed = unpackSeed(seedPacked.Load(s));
            if (seed.x > 1.5) continue;
            float d = dot(uv - seed, uv - seed);
            if (d < nearestDist) { nearestDist = d; nearestSeed = seed; }
        }
    if (nearestDist < 1e5) jfaPacked[tid] = packSeed(nearestSeed.x, nearestSeed.y);
    else jfaPacked[tid] = seedPacked.Load(int2(tid));
}

void jfaStepJfaToSeed(uint2 tid, uint2 dims, uint offset)
{
    float2 uv = (float2(tid) + 0.5) / float2(dims);
    float2 nearestSeed = float2(2.0, 2.0);
    float nearestDist = 1e6;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0) continue;
            int2 s = int2(tid) + int2(dx, dy) * int(offset);
            if (s.x < 0 || s.x >= int(dims.x) || s.y < 0 || s.y >= int(dims.y)) continue;
            float2 seed = unpackSeed(jfaPacked.Load(s));
            if (seed.x > 1.5) continue;
            float d = dot(uv - seed, uv - seed);
            if (d < nearestDist) { nearestDist = d; nearestSeed = seed; }
        }
    if (nearestDist < 1e5) seedPacked[tid] = packSeed(nearestSeed.x, nearestSeed.y);
    else seedPacked[tid] = jfaPacked.Load(int2(tid));
}

[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("seedPacked")]
void jfa0(uint2 tid: SV_DispatchThreadID) { uint2 d; seedPacked.GetDimensions(d.x, d.y); jfaStepSeedToJfa(tid, d, 128u); }
[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("seedPacked")]
void jfa1(uint2 tid: SV_DispatchThreadID) { uint2 d; seedPacked.GetDimensions(d.x, d.y); jfaStepJfaToSeed(tid, d, 64u); }
[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("seedPacked")]
void jfa2(uint2 tid: SV_DispatchThreadID) { uint2 d; seedPacked.GetDimensions(d.x, d.y); jfaStepSeedToJfa(tid, d, 32u); }
[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("seedPacked")]
void jfa3(uint2 tid: SV_DispatchThreadID) { uint2 d; seedPacked.GetDimensions(d.x, d.y); jfaStepJfaToSeed(tid, d, 16u); }
[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("seedPacked")]
void jfa4(uint2 tid: SV_DispatchThreadID) { uint2 d; seedPacked.GetDimensions(d.x, d.y); jfaStepSeedToJfa(tid, d, 8u); }
[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("seedPacked")]
void jfa5(uint2 tid: SV_DispatchThreadID) { uint2 d; seedPacked.GetDimensions(d.x, d.y); jfaStepJfaToSeed(tid, d, 4u); }
[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("seedPacked")]
void jfa6(uint2 tid: SV_DispatchThreadID) { uint2 d; seedPacked.GetDimensions(d.x, d.y); jfaStepSeedToJfa(tid, d, 2u); }
[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("seedPacked")]
void jfa7(uint2 tid: SV_DispatchThreadID) { uint2 d; seedPacked.GetDimensions(d.x, d.y); jfaStepJfaToSeed(tid, d, 1u); }

[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("distanceTex")]
void distancePass(uint2 tid: SV_DispatchThreadID)
{
    uint2 dims;
    distanceTex.GetDimensions(dims.x, dims.y);
    float2 uv = (float2(tid) + 0.5) / float2(dims);
    float2 seed = unpackSeed(seedPacked.Load(int2(tid)));
    float d = (seed.x > 1.5) ? 1.0 : length(uv - seed);
    distanceTex[tid] = d;
}

[shader("compute")]
[numthreads(8, 8, 1)]
[playground::CALL::SIZE_OF("radianceTex")]
void rcPass(uint2 tid: SV_DispatchThreadID)
{
    uint2 radDims, sceneDims;
    radianceTex.GetDimensions(radDims.x, radDims.y);
    sceneWrite.GetDimensions(sceneDims.x, sceneDims.y);
    float2 resolution = float2(sceneDims.x, sceneDims.y);
    float scale = min(resolution.x, resolution.y);
    float2 oneOverSize = 1.0 / resolution;
    float minStep = min(oneOverSize.x, oneOverSize.y) * 0.5;

    float2 probeCenter = (float2(tid) + 0.5) * basePixelsBetweenProbes;
    float angleStep = TAU / rayCount;
    float totalLum = 0.0;
    uint hits = 0;

    for (uint i = 0; i < uint(rayCount); i++)
    {
        float angle = (float(i) + 0.5) * angleStep;
        float2 rayDir = float2(cos(angle), -sin(angle));
        float2 rayStart = probeCenter + rayDir * 4.0;
        float rayLength = scale * 0.6;
        float2 rayUv = rayStart * oneOverSize;

        for (float dist = 0.0; dist < rayLength; )
        {
            if (rayUv.x < 0.0 || rayUv.x > 1.0 || rayUv.y < 0.0 || rayUv.y > 1.0) break;
            int2 tc = int2(rayUv * resolution);
            tc = clamp(tc, int2(0, 0), int2(sceneDims.x - 1, sceneDims.y - 1));
            float df = distanceTex.Load(tc);
            if (df <= minStep)
            {
                float4 c = unpackRGBA(sceneWrite.Load(tc));
                float lum = dot(c.rgb, float3(0.299, 0.587, 0.114)) * c.a;
                totalLum += lum;
                hits++;
                break;
            }
            float stepSize = df * scale * 0.5;
            dist += stepSize;
            rayUv += rayDir * (stepSize * oneOverSize);
        }
    }

    float lum = (hits > 0u) ? (totalLum / float(hits)) : 0.0;
    radianceTex[tid] = lum;
}

[shader("compute")]
[numthreads(THREADS, THREADS, 1)]
[playground::CALL::SIZE_OF("outputTexture")]
void overlay(uint2 tid: SV_DispatchThreadID)
{
    uint2 sceneDims, radDims;
    sceneWrite.GetDimensions(sceneDims.x, sceneDims.y);
    radianceTex.GetDimensions(radDims.x, radDims.y);
    float4 c = unpackRGBA(sceneWrite.Load(int2(tid)));
    float2 radUV = (float2(tid) + 0.5) * float2(radDims.x, radDims.y) / float2(sceneDims.x, sceneDims.y);
    int2 rp = int2(radUV);
    rp = clamp(rp, int2(0, 0), int2(radDims.x - 1, radDims.y - 1));
    float rad = radianceTex.Load(rp);
    float3 dark = float3(0.08, 0.06, 0.12);
    float3 radColor = dark + float3(rad, rad, rad);
    float3 final = (c.a > 0.01) ? c.rgb : radColor;
    drawPixel(tid, (int2 _) => float4(final, 1.0));
}
