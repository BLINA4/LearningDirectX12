#pragma once

#include <GameFramework/GameFramework.h>
#include <GameFramework/Window.h>

#include <dx12lib/Adapter.h>
#include <dx12lib/CommandList.h>
#include <dx12lib/CommandQueue.h>
#include <dx12lib/Device.h>
#include <dx12lib/Helpers.h>
#include <dx12lib/IndexBuffer.h>
#include <dx12lib/PipelineStateObject.h>
#include <dx12lib/RootSignature.h>
#include <dx12lib/SwapChain.h>
#include <dx12lib/VertexBuffer.h>

#include <dxc/dxcapi.h>

#include <spdlog/spdlog.h>

#include <shlwapi.h>  // for CommandLineToArgvW

#include <d3dcompiler.h>  // For D3DReadFileToBlob
#include <dxgidebug.h>    // For ReportLiveObjects.

#include <DirectXMath.h>  // For DirectX Math types.

using namespace Microsoft::WRL;
using namespace dx12lib;
using namespace DirectX;

ComPtr<IDxcResult> CompileShader( wchar_t* ShaderName );