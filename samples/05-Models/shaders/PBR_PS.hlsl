// clang-format off
struct PixelShaderInput
{
    float4 PositionVS  : POSITION;
    float3 NormalVS    : NORMAL;
    float3 TangentVS   : TANGENT;
    float3 BitangentVS : BITANGENT;
    float2 TexCoord    : TEXCOORD;
};

static const float PI = 3.14159265f;

struct Material
{
    float4 Diffuse;
    //------------------------------------ ( 16 bytes )
    float4 Specular;
    //------------------------------------ ( 16 bytes )
    float4 Emissive;
    //------------------------------------ ( 16 bytes )
    float4 Ambient;
    //------------------------------------ ( 16 bytes )
    float4 Reflectance;
    //------------------------------------ ( 16 bytes )
    float4 Albedo;
    //------------------------------------ ( 16 bytes )
    float4 Roughness;
    //------------------------------------ ( 16 bytes )
    float4 Metallic;
    //------------------------------------ ( 16 bytes )
    float4 AmbientOcclusion;
    //------------------------------------ ( 16 bytes )
    float Opacity;            // If Opacity < 1, then the material is transparent.
    float SpecularPower;
    float IndexOfRefraction;  // For transparent materials, IOR > 0.
    float BumpIntensity;      // When using bump textures (height maps) we need
                              // to scale the height values so the normals are visible.
    //------------------------------------ ( 16 bytes )
    bool  HasAmbientTexture;
    bool  HasEmissiveTexture;
    bool  HasDiffuseTexture;
    bool  HasSpecularTexture;
    //------------------------------------ ( 16 bytes )
    bool  HasAlbedoTexture;
    bool  HasRoughnessTexture;
    bool  HasMetallicTexture;
    bool  HasAmbientOcclusionTexture;
    //------------------------------------ ( 16 bytes )
    bool  HasSpecularPowerTexture;
    bool  HasNormalTexture;
    bool  HasBumpTexture;
    bool  HasOpacityTexture;
    //------------------------------------ ( 16 bytes )
    // Total:                              ( 16 * 13 = 198 bytes )
};

struct PointLight
{
    float4 PositionWS; // Light position in world space.
    //----------------------------------- (16 byte boundary)
    float4 PositionVS; // Light position in view space.
    //----------------------------------- (16 byte boundary)
    float4 Color;
    //----------------------------------- (16 byte boundary)
    float  Ambient;
    float  ConstantAttenuation;
    float  LinearAttenuation;
    float  QuadraticAttenuation;
    //----------------------------------- (16 byte boundary)
    // Total:                              16 * 4 = 64 bytes
};

struct SpotLight
{
    float4 PositionWS; // Light position in world space.
    //----------------------------------- (16 byte boundary)
    float4 PositionVS; // Light position in view space.
    //----------------------------------- (16 byte boundary)
    float4 DirectionWS; // Light direction in world space.
    //----------------------------------- (16 byte boundary)
    float4 DirectionVS; // Light direction in view space.
    //----------------------------------- (16 byte boundary)
    float4 Color;
    //----------------------------------- (16 byte boundary)
    float  Ambient;
    float  SpotAngle;
    float  ConstantAttenuation;
    float  LinearAttenuation;
    //----------------------------------- (16 byte boundary)
    float  QuadraticAttenuation;
    float3 Padding;
    //----------------------------------- (16 byte boundary)
    // Total:                              16 * 7 = 112 bytes
};

struct DirectionalLight
{
    float4 DirectionWS;  // Light direction in world space.
    //----------------------------------- (16 byte boundary)
    float4 DirectionVS;  // Light direction in view space.
    //----------------------------------- (16 byte boundary)
    float4 Color;
    //----------------------------------- (16 byte boundary)
    float Ambient;
    float3 Padding;
    //----------------------------------- (16 byte boundary)
    // Total:                              16 * 4 = 64 bytes
};

struct LightProperties
{
    uint NumPointLights;
    uint NumSpotLights;
    uint NumDirectionalLights;
};

struct LightResult
{
    float4 Diffuse;
    float4 Specular;
    float4 Ambient;
};

ConstantBuffer<LightProperties> LightPropertiesCB : register(b1);

StructuredBuffer<PointLight> PointLights : register(t0);
StructuredBuffer<SpotLight> SpotLights : register(t1);
StructuredBuffer<DirectionalLight> DirectionalLights : register(t2);

ConstantBuffer<Material> MaterialCB : register(b0, space1);

// Textures
Texture2D AmbientTexture            : register( t3 );
Texture2D EmissiveTexture           : register( t4 );
Texture2D DiffuseTexture            : register( t5 );
Texture2D SpecularTexture           : register( t6 );
Texture2D AlbedoTexture             : register( t7 );
Texture2D RoughnessTexture          : register( t8 );
Texture2D MetallicTexture           : register( t9 );
Texture2D AmbientOcclusionTexture   : register( t10 );
Texture2D SpecularPowerTexture      : register( t11 );
Texture2D NormalTexture             : register( t12 );
Texture2D BumpTexture               : register( t13 );
Texture2D OpacityTexture            : register( t14 );

SamplerState TextureSampler         : register(s0);

float3 LinearToSRGB( float3 x )
{
    // This is exactly the sRGB curve
    return x < 0.0031308 ? 12.92 * x : 1.055 * pow(abs(x), 1.0 / 2.4) - 0.055;

    // This is cheaper but nearly equivalent
    //return x < 0.0031308 ? 12.92 * x : 1.13005 * sqrt(abs(x - 0.00228)) - 0.13448 * x + 0.005719;
}

float3 SRGBToLinear( float3 x )
{
    return pow(x, 1.0 / 2.2);
}

float DoSpotCone( float3 spotDir, float3 L, float spotAngle )
{
    float minCos = cos(spotAngle);
    float maxCos = (minCos + 1.0f) / 2.0f;
    float cosAngle = dot(spotDir, -L);
    return smoothstep(minCos, maxCos, cosAngle);
}

/* Normal distribution function.
 * Arguments:
 *   - Normal vector
 *   - Halfway vector
 *   - Surface roughness
 */
float DistributionGGX( float3 N, float3 H, float a )
{
    float a2     = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
	
    float nom    = a2;
    float denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom        = PI * denom * denom;
	
    return nom / denom;
}

/* Geometry lightning approximation subfunction.
 * Arguments:
 *   - Normal * View value
 *   - Roughness value
 */
float GeometrySchlickGGX( float NdotV, float roughness )
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return nom / denom;
}

/* Geometry lightning approximation function.
 * Arguments:
 *   - Normal vector
 *   - View vector
 *   - Lightning vector
 *   - Surface roughness remap
 */  
float GeometrySmith( float3 N, float3 V, float3 L, float k )
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, k);
    float ggx2 = GeometrySchlickGGX(NdotL, k);
	
    return ggx1 * ggx2;
}

/* Fresnel-Schlick refraction approximation function.
 * Arguments:
 *   - Cosinus of light angle
 *   - Base reflectivity
 */
float3 fresnelSchlick( float cosTheta, float3 F0 )
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

/* Attenuation function.
 * Arguments:
 *   - constant factor
 *   - linear factor
 *   - quadratic factor
 *   - light distance (stands for intensity loss)
 */
float DoAttenuation( float c, float l, float q, float d )
{
    return 1.0f / (c + l * d + q * d * d);
}

/* Point light calculating function.
 * Arguments:
 *   - Vector to view position
 *   - Vector to point
 *   - Normal vector
 *   - Reflectance coefficient
 *   - Roughness value
 *   - Metallic value
 *   - Albedo value
 */
float3 DoPointLights( float3 V, float3 P, float3 N, float F0, float roughness, float metallic, float3 albedo )
{
    /* Reflectance equation */
    float3 Lo = float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < LightPropertiesCB.NumPointLights; ++i) 
    {
        /* Calculate per-light radiance */
        float3 L = normalize(PointLights[i].PositionVS - P);
        float3 H = normalize(V + L);
        float distance = length(PointLights[i].PositionVS - P);
        float attenuation = DoAttenuation(PointLights[i].ConstantAttenuation, PointLights[i].LinearAttenuation, PointLights[i].QuadraticAttenuation, distance);
        float3 radiance = PointLights[i].Color * attenuation;

        /* Cook-Torrance BRDF */
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);      
        float3 F  = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);
           
        float3 numerator    = NDF * G * F; 
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
        float3 specular = numerator / denominator;
        
        /* kS is equal to Fresnel */
        float3 kS = F;
        
        /* For energy conservation, the diffuse and specular light can't
         * be above 1.0 (unless the surface emits light); to preserve this
         * relationship the diffuse component (kD) should equal 1.0 - kS. */
        float3 kD = float3(1.0f, 1.0f, 1.0f) - kS;

        /* multiply kD by the inverse metalness such that only non-metals 
         * have diffuse lighting, or a linear blend if partly metal (pure metals
         * have no diffuse light). */
        kD *= 1.0 - metallic;	  

        /* Scale light by NdotL */
        float NdotL = max(dot(N, L), 0.0);        

        /* Add to outgoing radiance Lo */
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;  /* note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again */
    }

    return Lo;
}

/* Spot light calculating function.
 * Arguments:
 *   - Vector to view position
 *   - Vector to point
 *   - Normal vector
 *   - Reflectance coefficient
 *   - Roughness value
 *   - Metallic value
 *   - Albedo value
 */
float3 DoSpotLights( float3 V, float3 P, float3 N, float F0, float roughness, float metallic, float3 albedo )
{
    /* Reflectance equation */
    float3 Lo = float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < LightPropertiesCB.NumSpotLights; ++i) 
    {
        /* Calculate per-light radiance */
        float3 L = normalize(SpotLights[i].PositionVS - P);
        float3 H = normalize(V + L);
        float distance = length(SpotLights[i].PositionVS - P);
        float attenuation = DoAttenuation(SpotLights[i].ConstantAttenuation, SpotLights[i].LinearAttenuation, SpotLights[i].QuadraticAttenuation, distance);
        float3 radiance = SpotLights[i].Color * attenuation * DoSpotCone( SpotLights[i].DirectionVS.xyz, L, SpotLights[i].SpotAngle );

        /* Cook-Torrance BRDF */
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);      
        float3 F  = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);
           
        float3 numerator    = NDF * G * F; 
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
        float3 specular = numerator / denominator;
        
        /* kS is equal to Fresnel */
        float3 kS = F;
        
        /* For energy conservation, the diffuse and specular light can't
         * be above 1.0 (unless the surface emits light); to preserve this
         * relationship the diffuse component (kD) should equal 1.0 - kS. */
        float3 kD = float3(1.0f, 1.0f, 1.0f) - kS;

        /* multiply kD by the inverse metalness such that only non-metals 
         * have diffuse lighting, or a linear blend if partly metal (pure metals
         * have no diffuse light). */
        kD *= 1.0 - metallic;	  

        /* Scale light by NdotL */
        float NdotL = max(dot(N, L), 0.0);        

        /* Add to outgoing radiance Lo */
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;  /* note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again */
    }

    return Lo;
}

/* Directional light calculating function.
 * Arguments:
 *   - Vector to view position
 *   - Vector to point
 *   - Normal vector
 *   - Reflectance coefficient
 *   - Roughness value
 *   - Metallic value
 *   - Albedo value
 */
float3 DoDirectionalLights( float3 V, float3 P, float3 N, float F0, float roughness, float metallic, float3 albedo )
{
    /* Reflectance equation */
    float3 Lo = float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < LightPropertiesCB.NumDirectionalLights; ++i) 
    {
        /* Calculate per-light radiance */
        float3 L = normalize(-DirectionalLights[i].DirectionVS.xyz);
        float3 H = normalize(V + L);
        float3 radiance = DirectionalLights[i].Color;

        /* Cook-Torrance BRDF */
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);      
        float3 F  = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);
           
        float3 numerator    = NDF * G * F; 
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
        float3 specular = numerator / denominator;
        
        /* kS is equal to Fresnel */
        float3 kS = F;
        
        /* For energy conservation, the diffuse and specular light can't
         * be above 1.0 (unless the surface emits light); to preserve this
         * relationship the diffuse component (kD) should equal 1.0 - kS. */
        float3 kD = float3(1.0f, 1.0f, 1.0f) - kS;

        /* multiply kD by the inverse metalness such that only non-metals 
         * have diffuse lighting, or a linear blend if partly metal (pure metals
         * have no diffuse light). */
        kD *= 1.0 - metallic;	  

        /* Scale light by NdotL */
        float NdotL = max(dot(N, L), 0.0);        

        /* Add to outgoing radiance Lo */
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;  /* note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again */
    }

    return Lo;
}

/* Texture sampling function.
 * Arguments:
 *   - texture structure
 *   - texture coordinates
 *   - blending constant
 */
float4 SampleTexture( Texture2D t, float2 uv, float4 c )
{
    if (any(c.rgb))
    {
        c *= t.Sample(TextureSampler, uv);
    }
    else
    {
        c = t.Sample(TextureSampler, uv);
    }

    return c;
}

/* Sampling for metallic texture function.
 * Arguments:
 *   - texture structure
 *   - texture coordinates
 *   - blending constant
 */
float SampleMetallicTexture( Texture2D t, float2 uv, float c )
{
    if (c)
    {
        c *= t.Sample(TextureSampler, uv).b;
    }
    else
    {
        c = t.Sample(TextureSampler, uv).b;
    }

    return c;
}

/* Sampling for roughness texture function.
 * Arguments:
 *   - texture structure
 *   - texture coordinates
 *   - blending constant
 */
float SampleRoughnessTexture( Texture2D t, float2 uv, float c )
{
    if (c)
    {
        c *= t.Sample(TextureSampler, uv).g;
    }
    else
    {
        c = t.Sample(TextureSampler, uv).g;
    }

    return c;
}

/* Expand normal function.
 * Arguments:
 *   - Vector of expansion (factor)
 */
float3 ExpandNormal( float3 n )
{
    return n * 2.0f - 1.0f;
}

/* Normal mapping function.
 * Arguments:
 *   - 3x3 tangent bitangent normal matrix
 *   - Texture structure
 *   - Texture coordinates
 */
float3 DoNormalMapping( float3x3 TBN, Texture2D tex, float2 uv )
{
    float3 N = tex.Sample(TextureSampler, uv).xyz;
    N = ExpandNormal(N);

    // Transform normal from tangent space to view space.
    N = mul(N, TBN);
    return normalize(N);
}

/* Bump mapping function.
 * Arguments:
 *   - 3x3 tangent bitangent normal matrix
 *   - Texture structure
 *   - Texture coordinates
 *   - Scale factor
 */
float3 DoBumpMapping( float3x3 TBN, Texture2D tex, float2 uv, float bumpScale )
{
    // Sample the heightmap at the current texture coordinate.
    float height_00 = tex.Sample(TextureSampler, uv).r * bumpScale;
    // Sample the heightmap in the U texture coordinate direction.
    float height_10 = tex.Sample(TextureSampler, uv, int2(1, 0)).r * bumpScale;
    // Sample the heightmap in the V texture coordinate direction.
    float height_01 = tex.Sample(TextureSampler, uv, int2(0, 1)).r * bumpScale;

    float3 p_00 = {0, 0, height_00};
    float3 p_10 = {1, 0, height_10};
    float3 p_01 = {0, 1, height_01};

    // normal = tangent x bitangent
    float3 tangent = normalize(p_10 - p_00);
    float3 bitangent = normalize(p_01 - p_00);

    float3 normal = cross(tangent, bitangent);

    // Transform normal from tangent space to view space.
    normal = mul(normal, TBN);

    return normal;
}

/* Main shading function.
 * Arguments:
 *   - Pixel shader input structure
 *   - Fragment screen coordinates
 */
float4 main( PixelShaderInput IN, float4 ScreenCoord : SV_Position ) : SV_TARGET
{
    Material material = MaterialCB;

    // By default, use the alpha component of the diffuse color.
    float alpha = material.Diffuse.a;
    if (material.HasOpacityTexture) 
    {
        alpha = OpacityTexture.Sample(TextureSampler, IN.TexCoord.xy).r;
    }

    /* Discard the pixel if it is below a certain threshold.
     * if (alpha < 0.1f)
     * {
     *    discard;
     * }
     */

    float4 albedo = material.Albedo;
    float roughness = material.Roughness.r;
    float metallic = material.Metallic.r;
    float ao = material.AmbientOcclusion;
    float2 uv = IN.TexCoord.xy;

    if (material.HasAlbedoTexture)
    {
        albedo = SampleTexture(AlbedoTexture, uv, albedo);
    }
    if (material.HasMetallicTexture)
    {
        metallic = SampleMetallicTexture(MetallicTexture, uv, metallic);
        roughness = SampleRoughnessTexture(MetallicTexture, uv, roughness);
    }
    if (material.HasAmbientOcclusionTexture)
    {
        ao = SampleTexture(AmbientOcclusionTexture, uv, ao);
    }

    float3 N;
    // Normal mapping
    if (material.HasNormalTexture)
    {
        float3 tangent = normalize(IN.TangentVS);
        float3 bitangent = normalize(IN.BitangentVS);
        float3 normal = normalize(IN.NormalVS);

        float3x3 TBN = float3x3(tangent,
                                bitangent,
                                normal);

        N = DoNormalMapping(TBN, NormalTexture, uv);
    }
    else if (material.HasBumpTexture)
    {
        float3 tangent = normalize(IN.TangentVS);
        float3 bitangent = normalize(IN.BitangentVS);
        float3 normal = normalize(IN.NormalVS);

        float3x3 TBN = float3x3(tangent,
                                -bitangent,
                                normal);

        N = DoBumpMapping(TBN, BumpTexture, uv, material.BumpIntensity);
    }
    else
    {
        N = normalize(IN.NormalVS);
    }

    float3 P = normalize(IN.PositionVS);
    float3 V = -P;

    /* Calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
     * of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
     */
    float3 F0 = float3(0.04f, 0.04f, 0.04f); 
    F0 = lerp(F0, albedo, metallic);

    float3 Lo = float3(0.0f, 0.0f, 0.0f);
    
    Lo += DoPointLights(V, P, N, F0, roughness, metallic, albedo);
    Lo += DoSpotLights(V, P, N, F0, roughness, metallic, albedo);
    Lo += DoDirectionalLights(V, P, N, F0, roughness, metallic, albedo);
    
    /* Ambient lightning */
    float3 ambient = float3(0.03f, 0.03f, 0.03f) * albedo * ao;

    float3 color = ambient + Lo;

    /* HDR tonemapping */
    color = color / (color + float3(1.0f, 1.0f, 1.0f));
    
    /* Gamma correction */
    return float4(color, alpha * material.Opacity);
}