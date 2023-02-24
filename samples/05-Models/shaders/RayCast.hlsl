struct RayPayload
{
    bool  SkipShading;
    float RayHitT;
};

// Volatile part (can be split into its own CBV).
struct DynamicCB
{
    float4x4 cameraToWorld;
    float3   worldCameraPosition;
    uint     padding;
    float2   resolution;
};
    
#ifndef SINGLE
static const float FLT_MAX = asfloat( 0x7F7FFFFF );
#endif

RaytracingAccelerationStructure g_accel : register( t0 );

RWTexture2D<float4> g_screenOutput : register( u2 );

cbuffer HitShaderConstants : register( b0 )
{
    float3   SunDirection;
    float3   SunColor;
    float3   AmbientColor;
    float4   ShadowTexelSize;
    float4x4 ModelToShadow;
    uint     IsReflection;
    uint     UseShadowRays;
}

cbuffer b1 : register( b1 )
{
    DynamicCB g_dynamic;
};

inline void GenerateCameraRay( uint2 index, out float3 origin, out float3 direction )
{
    float2 xy        = index + 0.5;  // center in the middle of the pixel
    float2 screenPos = xy / g_dynamic.resolution * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates
    screenPos.y = -screenPos.y;

    // Unproject into a ray
    float4 unprojected = mul( g_dynamic.cameraToWorld, float4( screenPos, 0, 1 ) );
    float3 world       = unprojected.xyz / unprojected.w;
    origin             = g_dynamic.worldCameraPosition;
    direction          = normalize( world - origin );
}

[shader( "raygeneration" )] void RayGen() {
    float3 origin, direction;
    GenerateCameraRay( DispatchRaysIndex().xy, origin, direction );

    RayDesc    rayDesc = { origin, 0.0f, direction, FLT_MAX };
    RayPayload payload;
    payload.SkipShading = false;
    payload.RayHitT     = FLT_MAX;
    TraceRay( g_accel, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, payload );
}