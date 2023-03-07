#include "CommonDefine.h"
#include "../shaders/RayTracingHlslCompat.h"
#include "../../build_vs2022/bin/Debug/RayTracing.hlsl.h"

const wchar_t* c_hitGroupName         = L"MyHitGroup";
const wchar_t* c_raygenShaderName     = L"MyRaygenShader";
const wchar_t* c_closestHitShaderName = L"MyClosestHitShader";
const wchar_t* c_missShaderName       = L"MyMissShader";

// Library subobject names
// const wchar_t* c_globalRootSignatureName           = L"MyGlobalRootSignature";
// const wchar_t* c_localRootSignatureName            = L"MyLocalRootSignature";
// const wchar_t* c_localRootSignatureAssociationName = L"MyLocalRootSignatureAssociation";
// const wchar_t* c_shaderConfigName                  = L"MyShaderConfig";
// const wchar_t* c_pipelineConfigName                = L"MyPipelineConfig";

void OnUpdate( UpdateEventArgs& e );
void OnKeyPressed( KeyEventArgs& e );
void OnMouseWheel( MouseWheelEventArgs& e );
void OnResized( ResizeEventArgs& e );
void OnWindowClose( WindowCloseEventArgs& e );

std::shared_ptr<Device>              pDevice              = nullptr;
std::shared_ptr<Window>              pGameWindow          = nullptr;
std::shared_ptr<SwapChain>           pSwapChain           = nullptr;
std::shared_ptr<Texture>             pDepthTexture        = nullptr;
std::shared_ptr<VertexBuffer>        pVertexBuffer        = nullptr;
std::shared_ptr<IndexBuffer>         pIndexBuffer         = nullptr;
std::shared_ptr<RootSignature>       pRootSignature       = nullptr;

int width = 1920, height = 1080;

float FieldOfView = 45.0f;

// Pipeline state object for RT
ComPtr<ID3D12StateObject>            dxrStateObject;
ComPtr<ID3D12GraphicsCommandList4>   dxrCommandList;

// Buffer for constants
RayGenConstantBuffer                 rayGenCB;

// Descriptors
ComPtr<ID3D12DescriptorHeap>         descriptorHeap;
UINT                                 descriptorsAllocated;
UINT                                 descriptorSize;

// Root signatures
ComPtr<ID3D12RootSignature>          raytracingGlobalRootSignature;
ComPtr<ID3D12RootSignature>          raytracingLocalRootSignature;

// Shader tables
ComPtr<ID3D12Resource>               MissShaderTable;
ComPtr<ID3D12Resource>               HitGroupShaderTable;
ComPtr<ID3D12Resource>               RayGenShaderTable;

// Acceleration structure
ComPtr<ID3D12Resource>               accelerationStructure;
ComPtr<ID3D12Resource>               bottomLevelAccelerationStructure;
ComPtr<ID3D12Resource>               topLevelAccelerationStructure;

// Raytracing output
ComPtr<ID3D12Resource>               raytracingOutput;
D3D12_GPU_DESCRIPTOR_HANDLE          raytracingOutputResourceUAVGpuDescriptor;
UINT                                 raytracingOutputResourceUAVDescriptorHeapIndex;

Logger logger;

// Vertex data for a colored cube.
struct VertexPosColor
{
    XMFLOAT3 Position;
    XMFLOAT3 Color;
};

static WORD Indices[] = { 0, 1, 2, 2, 3, 0 };

float depthValue = 1.0;
float offset     = 0.5f;

static VertexPosColor Vertices[8] = 
{
    { XMFLOAT3( -offset, -offset, depthValue ), XMFLOAT3( 1.0f, 0.0f, 0.0f ) },
    { XMFLOAT3( -offset,  offset, depthValue ), XMFLOAT3( 1.0f, 0.0f, 0.0f ) },
    { XMFLOAT3(  offset,  offset, depthValue ), XMFLOAT3( 1.0f, 0.0f, 0.0f ) },
    { XMFLOAT3(  offset, -offset, depthValue ), XMFLOAT3( 1.0f, 0.0f, 0.0f ) }
};

#define SizeOfInUint32( obj ) ( ( sizeof( obj ) - 1 ) / sizeof( UINT32 ) + 1 )

namespace GlobalRootSignatureParams
{
    enum Value
    {
        OutputViewSlot = 0,
        AccelerationStructureSlot,
        Count
    };
}

namespace LocalRootSignatureParams
{
    enum Value
    {
        ViewportConstantSlot = 0,
        Count
    };
}

inline UINT Align( UINT size, UINT alignment )
{
    return ( size + ( alignment - 1 ) ) & ~( alignment - 1 );
}

// Shader record = {{Shader ID}, {RootArguments}}
class ShaderRecord
{
public:
    ShaderRecord( void* pShaderIdentifier, UINT shaderIdentifierSize )
    : shaderIdentifier( pShaderIdentifier, shaderIdentifierSize )
    {}

    ShaderRecord( void* pShaderIdentifier, UINT shaderIdentifierSize, void* pLocalRootArguments,
                  UINT localRootArgumentsSize )
    : shaderIdentifier( pShaderIdentifier, shaderIdentifierSize )
    , localRootArguments( pLocalRootArguments, localRootArgumentsSize )
    {}

    void CopyTo( void* dest ) const
    {
        uint8_t* byteDest = static_cast<uint8_t*>( dest );
        memcpy( byteDest, shaderIdentifier.ptr, shaderIdentifier.size );
        if ( localRootArguments.ptr )
        {
            memcpy( byteDest + shaderIdentifier.size, localRootArguments.ptr, localRootArguments.size );
        }
    }

    struct PointerWithSize
    {
        void* ptr;
        UINT  size;

        PointerWithSize()
        : ptr( nullptr )
        , size( 0 )
        {}
        PointerWithSize( void* _ptr, UINT _size )
        : ptr( _ptr )
        , size( _size ) {};
    };
    PointerWithSize shaderIdentifier;
    PointerWithSize localRootArguments;
};

class GpuUploadBuffer
{
public:
    ComPtr<ID3D12Resource> GetResource()
    {
        return m_resource;
    }

protected:
    ComPtr<ID3D12Resource> m_resource;

    GpuUploadBuffer() {}
    ~GpuUploadBuffer()
    {
        if ( m_resource.Get() )
        {
            m_resource->Unmap( 0, nullptr );
        }
    }

    void Allocate( ComPtr<ID3D12Device5> device, UINT bufferSize, LPCWSTR resourceName = nullptr )
    {
        auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD );

        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer( bufferSize );
        ThrowIfFailed( device->CreateCommittedResource( &uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                        IID_PPV_ARGS( &m_resource ) ) );
        m_resource->SetName( resourceName );
    }

    uint8_t* MapCpuWriteOnly()
    {
        uint8_t* mappedData;
        // We don't unmap this until the app closes. Keeping buffer mapped for the lifetime of the resource is okay.
        CD3DX12_RANGE readRange( 0, 0 );  // We do not intend to read from this resource on the CPU.
        ThrowIfFailed( m_resource->Map( 0, &readRange, reinterpret_cast<void**>( &mappedData ) ) );
        return mappedData;
    }
};

class ShaderTable : public GpuUploadBuffer
{
    uint8_t* m_mappedShaderRecords;
    UINT     m_shaderRecordSize;

    // Debug support
    std::wstring              m_name;
    std::vector<ShaderRecord> m_shaderRecords;

    ShaderTable() {}

public:
    ShaderTable( ComPtr<ID3D12Device5> device, UINT numShaderRecords, UINT shaderRecordSize, LPCWSTR resourceName = nullptr )
    : m_name( resourceName )
    {
        m_shaderRecordSize = Align( shaderRecordSize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT );
        m_shaderRecords.reserve( numShaderRecords );
        UINT bufferSize = numShaderRecords * m_shaderRecordSize;
        Allocate( device, bufferSize, resourceName );
        m_mappedShaderRecords = MapCpuWriteOnly();
    }

    void push_back( const ShaderRecord& shaderRecord )
    {
        // ThrowIfFalse( m_shaderRecords.size() < m_shaderRecords.capacity() );
        m_shaderRecords.push_back( shaderRecord );
        shaderRecord.CopyTo( m_mappedShaderRecords );
        m_mappedShaderRecords += m_shaderRecordSize;
    }

    UINT GetShaderRecordSize()
    {
        return m_shaderRecordSize;
    }
};

// Create raytracing device and command list.
void CreateRaytracingInterfaces( void )
{
    auto commandList = pDevice->GetCommandQueue().GetCommandList();

    ThrowIfFailed( commandList->GetD3D12CommandList()->QueryInterface( IID_PPV_ARGS( &dxrCommandList ) ) );
}

void SerializeAndCreateRaytracingRootSignature( D3D12_ROOT_SIGNATURE_DESC& desc, ComPtr<ID3D12RootSignature>* rootSig )
{
    auto             device = pDevice->GetD3D12Device();
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error;

    ThrowIfFailed( D3D12SerializeRootSignature( &desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error ) );
    ThrowIfFailed( device->CreateRootSignature( 1, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                IID_PPV_ARGS( &( *rootSig ) ) ) );
}

void CreateRootSignatures( void )
{
    // Global Root Signature
    // This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
    {
        CD3DX12_DESCRIPTOR_RANGE UAVDescriptor;
        UAVDescriptor.Init( D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0 );
        CD3DX12_ROOT_PARAMETER rootParameters[GlobalRootSignatureParams::Count];
        rootParameters[GlobalRootSignatureParams::OutputViewSlot].InitAsDescriptorTable( 1, &UAVDescriptor );
        rootParameters[GlobalRootSignatureParams::AccelerationStructureSlot].InitAsShaderResourceView( 0 );
        CD3DX12_ROOT_SIGNATURE_DESC globalRootSignatureDesc( ARRAYSIZE( rootParameters ), rootParameters );
        SerializeAndCreateRaytracingRootSignature( globalRootSignatureDesc, &raytracingGlobalRootSignature );
    }

    // Local Root Signature
    // This is a root signature that enables a shader to have unique arguments that come from shader tables.
    {
        CD3DX12_ROOT_PARAMETER rootParameters[LocalRootSignatureParams::Count];
        rootParameters[LocalRootSignatureParams::ViewportConstantSlot].InitAsConstants( SizeOfInUint32( rayGenCB ), 0, 0 );
        CD3DX12_ROOT_SIGNATURE_DESC localRootSignatureDesc( ARRAYSIZE( rootParameters ), rootParameters );
        localRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
        SerializeAndCreateRaytracingRootSignature( localRootSignatureDesc, &raytracingLocalRootSignature );
    }
}

// Local root signature and shader association
// This is a root signature that enables a shader to have unique arguments that come from shader tables.
void CreateLocalRootSignatureSubobjects( CD3DX12_STATE_OBJECT_DESC* raytracingPipeline )
{
    // Hit group and miss shaders in this sample are not using a local root signature and thus one is not associated
    // with them.

    // Local root signature to be used in a ray gen shader.
    {
        auto localRootSignature = raytracingPipeline->CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
        localRootSignature->SetRootSignature( raytracingLocalRootSignature.Get() );
        // Shader association
        auto rootSignatureAssociation =
            raytracingPipeline->CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
        rootSignatureAssociation->SetSubobjectToAssociate( *localRootSignature );
        rootSignatureAssociation->AddExport( c_raygenShaderName );
    }
}

void RTPSO( void )
{
    CD3DX12_STATE_OBJECT_DESC raytracingPipeline { D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };

    // DXIL library
    // This contains the shaders and their entrypoints for the state object.
    // Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
    auto                  lib     = raytracingPipeline.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE( (void*)g_pRaytracing, ARRAYSIZE( g_pRaytracing ) );
    lib->SetDXILLibrary( &libdxil );
    // Define which shader exports to surface from the library.
    // If no shader exports are defined for a DXIL library subobject, all shaders will be surfaced.
    // In this sample, this could be ommited for convenience since the sample uses all shaders in the library.
    {
        lib->DefineExport( c_raygenShaderName );
        lib->DefineExport( c_closestHitShaderName );
        lib->DefineExport( c_missShaderName );
    }

    // Triangle hit group
    // A hit group specifies closest hit, any hit and intersection shaders to be executed when a ray intersects the
    // geometry's triangle/AABB. In this sample, we only use triangle geometry with a closest hit shader, so others
    // are not set.
    auto hitGroup = raytracingPipeline.CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();
    hitGroup->SetClosestHitShaderImport( c_closestHitShaderName );
    hitGroup->SetHitGroupExport( c_hitGroupName );
    hitGroup->SetHitGroupType( D3D12_HIT_GROUP_TYPE_TRIANGLES );

    // Shader config
    // Defines the maximum sizes in bytes for the ray payload and attribute structure.
    auto shaderConfig  = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
    UINT payloadSize   = 4 * sizeof( float );  // float4 color
    UINT attributeSize = 2 * sizeof( float );  // float2 barycentrics
    shaderConfig->Config( payloadSize, attributeSize );

    // Local root signature and shader association
    CreateLocalRootSignatureSubobjects( &raytracingPipeline );
    // This is a root signature that enables a shader to have unique arguments that come from shader tables.

    // Global root signature
    // This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
    auto globalRootSignature = raytracingPipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
    globalRootSignature->SetRootSignature( raytracingGlobalRootSignature.Get() );

    // Pipeline config
    // Defines the maximum TraceRay() recursion depth.
    auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
    // PERFOMANCE TIP: Set max recursion depth as low as needed
    // as drivers may apply optimization strategies for low recursion depths.
    UINT maxRecursionDepth = 1;  // ~ primary rays only.
    pipelineConfig->Config( maxRecursionDepth );

    ThrowIfFailed(
        pDevice->GetD3D12Device()->CreateStateObject( raytracingPipeline, IID_PPV_ARGS( &dxrStateObject ) ) );
}

void CreateDescriptorHeap( void )
{
    auto device = pDevice->GetD3D12Device();

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    // Allocate a heap for a single descriptor:
    // 1 - raytracing output texture UAV
    descriptorHeapDesc.NumDescriptors = 1;
    descriptorHeapDesc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    descriptorHeapDesc.NodeMask       = 0;
    device->CreateDescriptorHeap( &descriptorHeapDesc, IID_PPV_ARGS( &descriptorHeap ) );
    NAME_D3D12_OBJECT( descriptorHeap );

    descriptorSize = device->GetDescriptorHandleIncrementSize( D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV );
}

UINT AllocateDescriptor( D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, UINT  descriptorIndexToUse )
{
    auto descriptorHeapCpuBase = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    if ( descriptorIndexToUse >= descriptorHeap->GetDesc().NumDescriptors )
    {
        descriptorIndexToUse = descriptorsAllocated++;
    }
    *cpuDescriptor = CD3DX12_CPU_DESCRIPTOR_HANDLE( descriptorHeapCpuBase, descriptorIndexToUse, descriptorSize );
    return descriptorIndexToUse;
}

// Create 2D output texture for raytracing.
void CreateRaytracingOutputResource()
{
    auto device           = pDevice->GetD3D12Device();
    // Use the render target from the swapchain.
    auto renderTarget     = pSwapChain->GetRenderTarget();
    auto backbufferFormat = renderTarget.GetRenderTargetFormats().RTFormats[0];

    // Create the output resource. The dimensions and format should match the swap-chain.
    auto uavDesc = CD3DX12_RESOURCE_DESC::Tex2D( backbufferFormat, width, height, 1, 1, 1, 0,
                                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS );

    auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT );
    ThrowIfFailed( device->CreateCommittedResource( &defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &uavDesc,
                                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                    IID_PPV_ARGS( &raytracingOutput ) ) );
    NAME_D3D12_OBJECT( raytracingOutput );

    D3D12_CPU_DESCRIPTOR_HANDLE uavDescriptorHandle;
    raytracingOutputResourceUAVDescriptorHeapIndex =
        AllocateDescriptor( &uavDescriptorHandle, raytracingOutputResourceUAVDescriptorHeapIndex );
    D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
    UAVDesc.ViewDimension                    = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView( raytracingOutput.Get(), nullptr, &UAVDesc, uavDescriptorHandle );
    raytracingOutputResourceUAVGpuDescriptor =
        CD3DX12_GPU_DESCRIPTOR_HANDLE( descriptorHeap->GetGPUDescriptorHandleForHeapStart(),
                                       raytracingOutputResourceUAVDescriptorHeapIndex, descriptorSize );
}

// Build geometry used in the sample.
void BuildGeometry( CommandQueue &queue )
{
    auto  commandList     = queue.GetCommandList();

    // Load vertex data:
    pVertexBuffer = commandList->CopyVertexBuffer( _countof( Vertices ), sizeof( VertexPosColor ), Vertices );

    // Load index data:
    pIndexBuffer = commandList->CopyIndexBuffer( _countof( Indices ), DXGI_FORMAT_R16_UINT, Indices );

    // Execute the command list to upload the resources to the GPU.
    queue.ExecuteCommandList( commandList );
}

// Build shader tables.
// This encapsulates all shader records - shaders and the arguments for their local root signatures.
void BuildShaderTables( void )
{
    auto device = pDevice->GetD3D12Device();

    void *rayGenShaderIdentifier;
    void *missShaderIdentifier;
    void *hitGroupShaderIdentifier;

    auto GetShaderIdentifiers = [&]( auto* stateObjectProperties ) {
        rayGenShaderIdentifier   = stateObjectProperties->GetShaderIdentifier( c_raygenShaderName );
        missShaderIdentifier     = stateObjectProperties->GetShaderIdentifier( c_missShaderName );
        hitGroupShaderIdentifier = stateObjectProperties->GetShaderIdentifier( c_hitGroupName );
    };

    // Get shader identifiers.
    UINT shaderIdentifierSize;
    {
        ComPtr<ID3D12StateObjectProperties> stateObjectProperties;
        ThrowIfFailed( dxrStateObject.As( &stateObjectProperties ) );
        GetShaderIdentifiers( stateObjectProperties.Get() );
        shaderIdentifierSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    }

    // Ray gen shader table
    {
        struct RootArguments
        {
            RayGenConstantBuffer cb;
        } rootArguments;
        rootArguments.cb = rayGenCB;

        UINT        numShaderRecords = 1;
        UINT        shaderRecordSize = shaderIdentifierSize + sizeof( rootArguments );
        ShaderTable rayGenShaderTable( device, numShaderRecords, shaderRecordSize, L"RayGenShaderTable" );
        rayGenShaderTable.push_back(
            ShaderRecord( rayGenShaderIdentifier, shaderIdentifierSize, &rootArguments, sizeof( rootArguments ) ) );
        RayGenShaderTable = rayGenShaderTable.GetResource();
    }

    // Miss shader table
    {
        UINT        numShaderRecords = 1;
        UINT        shaderRecordSize = shaderIdentifierSize;
        ShaderTable missShaderTable( device, numShaderRecords, shaderRecordSize, L"MissShaderTable" );
        missShaderTable.push_back( ShaderRecord( missShaderIdentifier, shaderIdentifierSize ) );
        MissShaderTable = missShaderTable.GetResource();
    }

    // Hit group shader table
    {
        UINT        numShaderRecords = 1;
        UINT        shaderRecordSize = shaderIdentifierSize;
        ShaderTable hitGroupShaderTable( device, numShaderRecords, shaderRecordSize, L"HitGroupShaderTable" );
        hitGroupShaderTable.push_back( ShaderRecord( hitGroupShaderIdentifier, shaderIdentifierSize ) );
        HitGroupShaderTable = hitGroupShaderTable.GetResource();
    }
}

inline void AllocateUAVBuffer( UINT64 bufferSize, ID3D12Resource **ppResource,
                               D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_COMMON,
                               const wchar_t        *resourceName         = nullptr )
{
    auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT );
    auto bufferDesc           = CD3DX12_RESOURCE_DESC::Buffer( bufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS );
    ThrowIfFailed( pDevice->GetD3D12Device()->CreateCommittedResource( 
        &uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        initialResourceState, nullptr, IID_PPV_ARGS( ppResource ) ) );
    
    if ( resourceName )
    {
        ( *ppResource )->SetName( resourceName );
    }
}

inline void AllocateUploadBuffer( void* pData, UINT64 datasize, ID3D12Resource** ppResource,
                                  const wchar_t* resourceName = nullptr )
{
    auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD );
    auto bufferDesc           = CD3DX12_RESOURCE_DESC::Buffer( datasize );
    ThrowIfFailed( pDevice->GetD3D12Device()->CreateCommittedResource( 
        &uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS( ppResource ) ) );
    if ( resourceName )
    {
        ( *ppResource )->SetName( resourceName );
    }
    void* pMappedData;
    ( *ppResource )->Map( 0, nullptr, &pMappedData );
    memcpy( pMappedData, pData, datasize );
    ( *ppResource )->Unmap( 0, nullptr );
}

// Build acceleration structures needed for raytracing.
void BuildAccelerationStructures( void )
{
    auto device        = pDevice->GetD3D12Device();
    auto& commandQueue = pDevice->GetCommandQueue( D3D12_COMMAND_LIST_TYPE_COPY );
    auto  commandList  = commandQueue.GetCommandList();

    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
    geometryDesc.Type                           = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometryDesc.Triangles.IndexBuffer          = pIndexBuffer->GetD3D12Resource()->GetGPUVirtualAddress();
    geometryDesc.Triangles.IndexCount           = static_cast<UINT>( pIndexBuffer->GetD3D12ResourceDesc().Width ) / sizeof( WORD );
    geometryDesc.Triangles.IndexFormat          = DXGI_FORMAT_R16_UINT;
    geometryDesc.Triangles.Transform3x4         = 0;
    geometryDesc.Triangles.VertexFormat         = DXGI_FORMAT_R32G32B32_FLOAT;
    geometryDesc.Triangles.VertexCount = static_cast<UINT>( pVertexBuffer->GetD3D12ResourceDesc().Width ) / sizeof( VertexPosColor );
    geometryDesc.Triangles.VertexBuffer.StartAddress  = pVertexBuffer->GetD3D12Resource()->GetGPUVirtualAddress();
    geometryDesc.Triangles.VertexBuffer.StrideInBytes = sizeof( VertexPosColor );

    // Mark the geometry as opaque.
    // PERFORMANCE TIP: mark geometry as opaque whenever applicable as it can enable important ray processing
    // optimizations. Note: When rays encounter opaque geometry an any hit shader will not be executed whether it is
    // present or not.
    geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

    // Get required sizes for an acceleration structure.
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS topLevelInputs = {};
    topLevelInputs.DescsLayout                                          = D3D12_ELEMENTS_LAYOUT_ARRAY;
    topLevelInputs.Flags                                                = buildFlags;
    topLevelInputs.NumDescs                                             = 1;
    topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo = {};
    pDevice->GetD3D12Device()->GetRaytracingAccelerationStructurePrebuildInfo( &topLevelInputs, &topLevelPrebuildInfo );
    //ThrowIfFalse( topLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0 );

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bottomLevelPrebuildInfo = {};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS  bottomLevelInputs       = topLevelInputs;
    bottomLevelInputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottomLevelInputs.pGeometryDescs = &geometryDesc;
    pDevice->GetD3D12Device()->GetRaytracingAccelerationStructurePrebuildInfo( &bottomLevelInputs, &bottomLevelPrebuildInfo );
    //ThrowIfFalse( bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes > 0 );

    ComPtr<ID3D12Resource> scratchResource;
    AllocateUAVBuffer(
        std::max( topLevelPrebuildInfo.ScratchDataSizeInBytes, bottomLevelPrebuildInfo.ScratchDataSizeInBytes ),
        &scratchResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"ScratchResource" );

    // Allocate resources for acceleration structures.
    // Acceleration structures can only be placed in resources that are created in the default heap (or custom heap
    // equivalent). Default heap is OK since the application doesn’t need CPU read/write access to them. The resources
    // that will contain acceleration structures must be created in the state
    // D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, and must have resource flag
    // D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS. The ALLOW_UNORDERED_ACCESS requirement simply acknowledges both:
    //  - the system will be doing this type of access in its implementation of acceleration structure builds behind the
    //  scenes.
    //  - from the app point of view, synchronization of writes/reads to acceleration structures is accomplished using
    //  UAV barriers.
    {
        D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;

        AllocateUAVBuffer( bottomLevelPrebuildInfo.ResultDataMaxSizeInBytes,
                           &bottomLevelAccelerationStructure, initialResourceState,
                           L"BottomLevelAccelerationStructure" );
        AllocateUAVBuffer( topLevelPrebuildInfo.ResultDataMaxSizeInBytes, &topLevelAccelerationStructure,
                           initialResourceState, L"TopLevelAccelerationStructure" );
    }

    // Create an instance desc for the bottom-level acceleration structure.
    ComPtr<ID3D12Resource>         instanceDescs;
    D3D12_RAYTRACING_INSTANCE_DESC instanceDesc = {};
    instanceDesc.Transform[0][0] = instanceDesc.Transform[1][1] = instanceDesc.Transform[2][2] = 1;
    instanceDesc.InstanceMask                                                                  = 1;
    instanceDesc.AccelerationStructure = bottomLevelAccelerationStructure->GetGPUVirtualAddress();
    AllocateUploadBuffer( &instanceDesc, sizeof( instanceDesc ), &instanceDescs, L"InstanceDescs" );

    // Bottom Level Acceleration Structure desc
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomLevelBuildDesc = {};
    {
        bottomLevelBuildDesc.Inputs                           = bottomLevelInputs;
        bottomLevelBuildDesc.ScratchAccelerationStructureData = scratchResource->GetGPUVirtualAddress();
        bottomLevelBuildDesc.DestAccelerationStructureData = bottomLevelAccelerationStructure->GetGPUVirtualAddress();
    }

    // Top Level Acceleration Structure desc
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc = {};
    {
        topLevelInputs.InstanceDescs                       = instanceDescs->GetGPUVirtualAddress();
        topLevelBuildDesc.Inputs                           = topLevelInputs;
        topLevelBuildDesc.DestAccelerationStructureData    = topLevelAccelerationStructure->GetGPUVirtualAddress();
        topLevelBuildDesc.ScratchAccelerationStructureData = scratchResource->GetGPUVirtualAddress();
    }

    auto BuildAccelerationStructure = [&]( auto* raytracingCommandList ) {
        raytracingCommandList->BuildRaytracingAccelerationStructure( &bottomLevelBuildDesc, 0, nullptr );
        commandList->UAVBarrier( bottomLevelAccelerationStructure.Get() );
        raytracingCommandList->BuildRaytracingAccelerationStructure( &topLevelBuildDesc, 0, nullptr );
    };

    // Build acceleration structure.
    BuildAccelerationStructure( dxrCommandList.Get() );

    // Kick off acceleration structure construction.
    commandQueue.ExecuteCommandList( commandList );
}

void DoRaytracing( std::shared_ptr<dx12lib::CommandList> commandList )
{
    auto DispatchRays = [&]( auto* list, auto* stateObject, auto* dispatchDesc ) {
        // Since each shader table has only one shader record, the stride is same as the size.
        dispatchDesc->HitGroupTable.StartAddress             = HitGroupShaderTable->GetGPUVirtualAddress();
        dispatchDesc->HitGroupTable.SizeInBytes              = HitGroupShaderTable->GetDesc().Width;
        dispatchDesc->HitGroupTable.StrideInBytes            = dispatchDesc->HitGroupTable.SizeInBytes;
        dispatchDesc->MissShaderTable.StartAddress           = MissShaderTable->GetGPUVirtualAddress();
        dispatchDesc->MissShaderTable.SizeInBytes            = MissShaderTable->GetDesc().Width;
        dispatchDesc->MissShaderTable.StrideInBytes          = dispatchDesc->MissShaderTable.SizeInBytes;
        dispatchDesc->RayGenerationShaderRecord.StartAddress = RayGenShaderTable->GetGPUVirtualAddress();
        dispatchDesc->RayGenerationShaderRecord.SizeInBytes  = RayGenShaderTable->GetDesc().Width;
        dispatchDesc->Width                                  = width;
        dispatchDesc->Height                                 = height;
        dispatchDesc->Depth                                  = 1;
        list->SetPipelineState1( stateObject );
        list->DispatchRays( dispatchDesc );
    };

    commandList->GetD3D12CommandList()->SetComputeRootSignature( raytracingGlobalRootSignature.Get() );

    // Bind the heaps, acceleration structure and dispatch rays.
    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
    commandList->GetD3D12CommandList()->SetDescriptorHeaps( 1, descriptorHeap.GetAddressOf() );
    commandList->GetD3D12CommandList()->SetComputeRootDescriptorTable(    GlobalRootSignatureParams::OutputViewSlot,
                                                                          raytracingOutputResourceUAVGpuDescriptor );
    commandList->GetD3D12CommandList()->SetComputeRootShaderResourceView( GlobalRootSignatureParams::AccelerationStructureSlot,
                                                                          topLevelAccelerationStructure->GetGPUVirtualAddress() );
    DispatchRays( commandList->GetD3D12CommandList().Get(), dxrStateObject.Get(), &dispatchDesc );
}

int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow )
{
    int retCode = 0;

#if defined( _DEBUG )
    Device::EnableDebugLayer();
#endif

    // Set the working directory to the path of the executable.
    WCHAR   path[MAX_PATH];
    HMODULE hModule = ::GetModuleHandleW( NULL );
    if ( ::GetModuleFileNameW( hModule, path, MAX_PATH ) > 0 )
    {
        ::PathRemoveFileSpecW( path );
        ::SetCurrentDirectoryW( path );
    }

    auto& gf = GameFramework::Create( hInstance );
    {
        // Create a logger for logging messages.
        logger = gf.CreateLogger( "Cube" );

        // Create a GPU device using the default adapter selection.
        pDevice = Device::Create();

        auto description = pDevice->GetDescription();
        logger->info( L"Device Created: {}", description );

        // Use a copy queue to copy GPU resources.
        auto& commandQueue = pDevice->GetCommandQueue();
        auto  commandList  = commandQueue.GetCommandList();

        BuildGeometry( commandQueue );

        // Create a window:
        pGameWindow = gf.CreateWindow( L"Cube", width, height );

        // Create a swapchain for the window
        pSwapChain = pDevice->CreateSwapChain( pGameWindow->GetWindowHandle() );
        pSwapChain->SetVSync( false );

        // Register events.
        pGameWindow->KeyPressed += &OnKeyPressed;
        pGameWindow->MouseWheel += &OnMouseWheel;
        pGameWindow->Resize += &OnResized;
        pGameWindow->Update += &OnUpdate;
        pGameWindow->Close += &OnWindowClose;

        CreateRaytracingInterfaces();
        CreateRootSignatures();
        RTPSO();
        CreateDescriptorHeap();
        BuildAccelerationStructures();
        BuildShaderTables();
        CreateRaytracingOutputResource();

        // Make sure the index/vertex buffers are loaded before rendering the first frame.
        commandQueue.Flush();

        pGameWindow->Show();

        // Start the game loop.
        retCode = GameFramework::Get().Run();

        // Release globals.
        pIndexBuffer.reset();
        pVertexBuffer.reset();
        //pPipelineStateObject.reset();
        dxrStateObject.Reset();
        //pRootSignature.reset();
        raytracingGlobalRootSignature.Reset();
        raytracingLocalRootSignature.Reset();
        pDepthTexture.reset();
        pDevice.reset();
        pSwapChain.reset();
        pGameWindow.reset();
    }
    // Destroy game framework resource.
    GameFramework::Destroy();

    atexit( &Device::ReportLiveObjects );

    return retCode;
}

void OnUpdate( UpdateEventArgs& e )
{
    static uint64_t frameCount = 0;
    static double   totalTime  = 0.0;

    totalTime += e.DeltaTime;
    frameCount++;

    if ( totalTime > 1.0 )
    {
        auto fps   = frameCount / totalTime;
        frameCount = 0;
        totalTime -= 1.0;

        logger->info( "FPS: {:.7}", fps );

        wchar_t buffer[256];
        ::swprintf_s( buffer, L"Cube [FPS: %f]", fps );
        pGameWindow->SetWindowTitle( buffer );
    }

    // Use the render target from the swapchain.
    auto renderTarget = pSwapChain->GetRenderTarget();
    // Set the render target (with the depth texture).
    renderTarget.AttachTexture( AttachmentPoint::DepthStencil, pDepthTexture );

    auto viewport = renderTarget.GetViewport();

    // Update the model matrix.
    float          angle        = static_cast<float>( e.TotalTime * 90.0 );
    const XMVECTOR rotationAxis = XMVectorSet( 0, 1, 1, 0 );
    XMMATRIX       modelMatrix  = XMMatrixRotationAxis( rotationAxis, XMConvertToRadians( angle ) );

    // Update the view matrix.
    const XMVECTOR eyePosition = XMVectorSet( 0, 0, -10, 1 );
    const XMVECTOR focusPoint  = XMVectorSet( 0, 0, 0, 1 );
    const XMVECTOR upDirection = XMVectorSet( 0, 1, 0, 0 );
    XMMATRIX       viewMatrix  = XMMatrixLookAtLH( eyePosition, focusPoint, upDirection );

    // Update the projection matrix.
    float    aspectRatio = viewport.Width / viewport.Height;
    XMMATRIX projectionMatrix =
        XMMatrixPerspectiveFovLH( XMConvertToRadians( FieldOfView ), aspectRatio, 0.1f, 100.0f );
    XMMATRIX mvpMatrix = XMMatrixMultiply( modelMatrix, viewMatrix );
    mvpMatrix          = XMMatrixMultiply( mvpMatrix, projectionMatrix );

    auto& commandQueue = pDevice->GetCommandQueue( D3D12_COMMAND_LIST_TYPE_DIRECT );
    auto  commandList  = commandQueue.GetCommandList();

    // Set the root signatures.
    //commandList->SetGraphicsRootSignature( raytracingGlobalRootSignature );

    // Set root parameters
    //commandList->SetGraphics32BitConstants( 0, mvpMatrix );

    // Clear the color and depth-stencil textures.
    const FLOAT clearColor[] = { 0.4f, 0.6f, 0.9f, 1.0f };
    commandList->ClearTexture( renderTarget.GetTexture( AttachmentPoint::Color0 ), clearColor );
    commandList->ClearDepthStencilTexture( pDepthTexture, D3D12_CLEAR_FLAG_DEPTH );

    commandList->SetRenderTarget( renderTarget );
    commandList->SetViewport( renderTarget.GetViewport() );
    commandList->SetScissorRect( CD3DX12_RECT( 0, 0, LONG_MAX, LONG_MAX ) );
    
    DoRaytracing( commandList );

    commandQueue.ExecuteCommandList( commandList );

    // Present the image to the window.
    pSwapChain->Present();
}

void OnKeyPressed( KeyEventArgs& e )
{
    switch ( e.Key )
    {
    case KeyCode::V:
        pSwapChain->ToggleVSync();
        break;
    case KeyCode::Escape:
        // Stop the application if the Escape key is pressed.
        GameFramework::Get().Stop();
        break;
    case KeyCode::Enter:
        if ( e.Alt )
        {
            [[fallthrough]];
        case KeyCode::F11:
            pGameWindow->ToggleFullscreen();
            break;
        }
    }
}

void OnMouseWheel( MouseWheelEventArgs& e )
{
    FieldOfView -= e.WheelDelta;
    FieldOfView = std::clamp( FieldOfView, 12.0f, 90.0f );

    logger->info( "Field of View: {}", FieldOfView );
}

void OnResized( ResizeEventArgs& e )
{
    width = e.Width, height = e.Height;
    logger->info( "Window Resize: {}, {}", e.Width, e.Height );
    GameFramework::Get().SetDisplaySize( e.Width, e.Height );

    // Flush any pending commands before resizing resources.
    pDevice->Flush();

    // Resize the swap chain.
    pSwapChain->Resize( e.Width, e.Height );

    // Resize the depth texture.
    auto depthTextureDesc = CD3DX12_RESOURCE_DESC::Tex2D( DXGI_FORMAT_D32_FLOAT, e.Width, e.Height );
    // Must be set on textures that will be used as a depth-stencil buffer.
    depthTextureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    // Specify optimized clear values for the depth buffer.
    D3D12_CLEAR_VALUE optimizedClearValue = {};
    optimizedClearValue.Format            = DXGI_FORMAT_D32_FLOAT;
    optimizedClearValue.DepthStencil      = { 1.0F, 0 };

    pDepthTexture = pDevice->CreateTexture( depthTextureDesc, &optimizedClearValue );
}

void OnWindowClose( WindowCloseEventArgs& e )
{
    // Stop the application if the window is closed.
    GameFramework::Get().Stop();
}
