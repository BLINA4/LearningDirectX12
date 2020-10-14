#include "DX12LibPCH.h"

#include <dx12lib/SwapChain.h>

#include <dx12lib/Adapter.h>
#include <dx12lib/CommandList.h>
#include <dx12lib/CommandQueue.h>
#include <dx12lib/Device.h>
#include <dx12lib/GUI.h>
#include <dx12lib/RenderTarget.h>
#include <dx12lib/ResourceStateTracker.h>
#include <dx12lib/Texture.h>

using namespace dx12lib;

SwapChain::SwapChain( std::shared_ptr<Device> device, HWND hWnd )
: m_Device( device )
, m_CommandQueue( device->GetCommandQueue( D3D12_COMMAND_LIST_TYPE_DIRECT ) )
, m_hWnd( hWnd )
, m_FenceValues { 0 }
, m_Width( 0u )
, m_Height( 0u )
, m_VSync( true )
, m_TearingSupported( false )
, m_Fullscreen( false )
{
    assert( hWnd );    // Must be a valid window handle!
    assert( device );  // Must be a valid device!

    // Query the direct command queue from the device.
    // This is required to create the swapchain.
    auto d3d12CommandQueue = m_CommandQueue->GetD3D12CommandQueue();

    // Query the factory from the adapter that was used to create the device.
    auto adapter     = device->GetAdapter();
    auto dxgiAdapter = adapter->GetDXGIAdapter();

    // Get the factory that was used to create the adapter.
    ComPtr<IDXGIFactory>  dxgiFactory;
    ComPtr<IDXGIFactory5> dxgiFactory5;
    ThrowIfFailed( dxgiAdapter->GetParent( IID_PPV_ARGS( &dxgiFactory ) ) );
    // Now get the DXGIFactory5 so I can use the IDXGIFactory5::CheckFeatureSupport method.
    ThrowIfFailed( dxgiFactory.As( &dxgiFactory5 ) );

    //    UINT                  createFactoryFlags = 0;
    //#if defined( _DEBUG )
    //    createFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
    //#endif
    //
    //    ThrowIfFailed( CreateDXGIFactory2( createFactoryFlags, IID_PPV_ARGS( &dxgiFactory5 ) ) );

    // Check for tearing support.
    BOOL allowTearing = FALSE;
    if ( SUCCEEDED(
             dxgiFactory5->CheckFeatureSupport( DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof( BOOL ) ) ) )
    {
        m_TearingSupported = ( allowTearing == TRUE );
    }

    // Query the windows client width and height.
    RECT windowRect;
    ::GetClientRect( hWnd, &windowRect );

    m_Width  = windowRect.right - windowRect.left;
    m_Height = windowRect.bottom - windowRect.top;

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.Width                 = m_Width;
    swapChainDesc.Height                = m_Height;
    swapChainDesc.Format                = DXGI_FORMAT_R8G8B8A8_UNORM;  // TODO: Check for HDR support.
    swapChainDesc.Stereo                = FALSE;
    swapChainDesc.SampleDesc            = { 1, 0 };
    swapChainDesc.BufferUsage           = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount           = BufferCount;
    swapChainDesc.Scaling               = DXGI_SCALING_STRETCH;
    swapChainDesc.SwapEffect            = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.AlphaMode             = DXGI_ALPHA_MODE_UNSPECIFIED;
    // It is recommended to always allow tearing if tearing support is available.
    swapChainDesc.Flags = m_TearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    swapChainDesc.Flags |= DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

    // Now create the swap chain.
    ComPtr<IDXGISwapChain1> dxgiSwapChain1;
    ThrowIfFailed( dxgiFactory5->CreateSwapChainForHwnd( d3d12CommandQueue.Get(), m_hWnd, &swapChainDesc, nullptr,
                                                         nullptr, &dxgiSwapChain1 ) );

    // Cast to swapchain4
    ThrowIfFailed( dxgiSwapChain1.As( &m_dxgiSwapChain ) );

    // Disable the Alt+Enter fullscreen toggle feature. Switching to fullscreen
    // will be handled manually.
    ThrowIfFailed( dxgiFactory5->MakeWindowAssociation( m_hWnd, DXGI_MWA_NO_ALT_ENTER ) );

    // Initialize the current back buffer index.
    m_CurrentBackBufferIndex = m_dxgiSwapChain->GetCurrentBackBufferIndex();

    // Set maximum frame latency to reduce input latency.
    m_dxgiSwapChain->SetMaximumFrameLatency( 1 );
    // Get the SwapChain's waitable object.
    m_hFrameLatencyWaitableObject = m_dxgiSwapChain->GetFrameLatencyWaitableObject();

    UpdateRenderTargetViews();

    m_GUI = std::make_shared<GUI>( device, m_hWnd );
    m_GUI->NewFrame();
}

SwapChain::~SwapChain()
{
    // Close the SwapChain's waitable event handle?? I don't think that's necessary!?
    //::CloseHandle( m_hFrameLatencyWaitableObject );
}

void SwapChain::SetFullscreen( bool fullscreen )
{
    if ( m_Fullscreen != fullscreen )
    {
        m_Fullscreen = fullscreen;
        // TODO:
        //        m_dxgiSwapChain->SetFullscreenState()
    }
}

void dx12lib::SwapChain::WaitForSwapChain()
{
    DWORD result = ::WaitForSingleObjectEx( m_hFrameLatencyWaitableObject, 1000,
                                            TRUE );  // Wait for 1 second (should never have to wait that long...)
}

void dx12lib::SwapChain::Resize( uint32_t width, uint32_t height )
{
    if ( m_Width != width || m_Height != height )
    {
        m_Width  = std::max( 1u, width );
        m_Height = std::max( 1u, height );

        auto device = m_Device.lock();
        device->Flush();

        // Release all references to back buffer textures.
        m_RenderTarget.AttachTexture( AttachmentPoint::Color0, nullptr );
        for ( UINT i = 0; i < BufferCount; ++i )
        {
            ResourceStateTracker::RemoveGlobalResourceState( m_BackBufferTextures[i]->GetD3D12Resource().Get() );
            m_BackBufferTextures[i].reset();
        }

        DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
        ThrowIfFailed( m_dxgiSwapChain->GetDesc( &swapChainDesc ) );
        ThrowIfFailed( m_dxgiSwapChain->ResizeBuffers( BufferCount, m_Width, m_Height, swapChainDesc.BufferDesc.Format,
                                                       swapChainDesc.Flags ) );

        m_CurrentBackBufferIndex = m_dxgiSwapChain->GetCurrentBackBufferIndex();

        UpdateRenderTargetViews();
    }
}

const dx12lib::RenderTarget& dx12lib::SwapChain::GetRenderTarget() const
{
    m_RenderTarget.AttachTexture( AttachmentPoint::Color0, m_BackBufferTextures[m_CurrentBackBufferIndex] );
    return m_RenderTarget;
}

UINT dx12lib::SwapChain::Present( std::shared_ptr<Texture> texture )
{
    auto device      = m_Device.lock();
    auto commandList = m_CommandQueue->GetCommandList();

    auto backBuffer = m_BackBufferTextures[m_CurrentBackBufferIndex];

    if ( texture )
    {
        if ( texture->GetD3D12ResourceDesc().SampleDesc.Count > 1 )
        {
            commandList->ResolveSubresource( backBuffer, texture );
        }
        else
        {
            commandList->CopyResource( backBuffer, texture );
        }
    }

     RenderTarget renderTarget;
     renderTarget.AttachTexture( AttachmentPoint::Color0, backBuffer );
     m_GUI->Render( commandList, renderTarget );

    commandList->TransitionBarrier( backBuffer, D3D12_RESOURCE_STATE_PRESENT );
    m_CommandQueue->ExecuteCommandList( commandList );

    UINT syncInterval = m_VSync ? 1 : 0;
    UINT presentFlags = m_TearingSupported && !m_Fullscreen && !m_VSync ? DXGI_PRESENT_ALLOW_TEARING : 0;
    ThrowIfFailed( m_dxgiSwapChain->Present( syncInterval, presentFlags ) );

    m_FenceValues[m_CurrentBackBufferIndex] = m_CommandQueue->Signal();

    m_CurrentBackBufferIndex = m_dxgiSwapChain->GetCurrentBackBufferIndex();

    auto fenceValue = m_FenceValues[m_CurrentBackBufferIndex];
    m_CommandQueue->WaitForFenceValue( fenceValue );

    device->ReleaseStaleDescriptors( fenceValue );

    m_GUI->NewFrame();

    return m_CurrentBackBufferIndex;
}

void dx12lib::SwapChain::UpdateRenderTargetViews()
{
    auto device = m_Device.lock();

    for ( UINT i = 0; i < BufferCount; ++i )
    {
        ComPtr<ID3D12Resource> backBuffer;
        ThrowIfFailed( m_dxgiSwapChain->GetBuffer( i, IID_PPV_ARGS( &backBuffer ) ) );

        ResourceStateTracker::AddGlobalResourceState( backBuffer.Get(), D3D12_RESOURCE_STATE_COMMON );

        m_BackBufferTextures[i] = device->CreateTexture( backBuffer, nullptr, TextureUsage::RenderTarget );

        // Set the names for the backbuffer textures.
        // Useful for debugging.
        m_BackBufferTextures[i]->SetName( L"Backbuffer[" + std::to_wstring( i ) + L"]" );
    }
}