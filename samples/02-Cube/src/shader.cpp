#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include "../../02-Cube/src/CommonDefine.h"

std::string ReadShaderFile( wchar_t *FileName ) 
{
    std::ifstream inFile;
    inFile.open( FileName );  // open the input file

    std::stringstream strStream;
    strStream << inFile.rdbuf();  // read the file
    
    return strStream.str();       // str holds the content of the file
}

ComPtr<IDxcResult> CompileShader( wchar_t* ShaderName )
{
    ComPtr<IDxcUtils> pUtils;
    DxcCreateInstance( CLSID_DxcUtils, IID_PPV_ARGS( pUtils.GetAddressOf() ) );

    std::string pShaderSource = ReadShaderFile( ShaderName );

    ComPtr<IDxcBlobEncoding> pSource;
    pUtils->CreateBlob( pShaderSource.c_str(), pShaderSource.size(), CP_UTF8, pSource.GetAddressOf() );

    std::vector<LPWSTR> arguments;
    //-E for the entry point (eg. PSMain)
    //arguments.push_back( L"-E" );
    //arguments.push_back( L"main" );

    //-T for the target profile (eg. ps_6_2)
    arguments.push_back( L"-T" );
    arguments.push_back( L"lib_6_7" );

    // Strip reflection data and pdbs (see later)
    arguments.push_back( L"-Qstrip_debug" );
    arguments.push_back( L"-Qstrip_reflect" );

    arguments.push_back( DXC_ARG_WARNINGS_ARE_ERRORS );    //-WX
    arguments.push_back( DXC_ARG_DEBUG );                  //-Zi
    arguments.push_back( DXC_ARG_PACK_MATRIX_ROW_MAJOR );  //-Zp

    /*
    for ( const std::wstring& define: defines )
    {
        arguments.push_back( L"-D" );
        arguments.push_back( define.c_str() );
    }
    */

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr      = pSource->GetBufferPointer();
    sourceBuffer.Size     = pSource->GetBufferSize();
    sourceBuffer.Encoding = 0;

    ComPtr<IDxcCompiler3> pCompiler;
    DxcCreateInstance( CLSID_DxcCompiler, IID_PPV_ARGS( pCompiler.GetAddressOf() ) );

    ComPtr<IDxcResult> pCompileResult;
    HRESULT            HR = pCompiler->Compile( &sourceBuffer, (LPCWSTR *)&arguments[0], (UINT32)arguments.size(), nullptr,
                            IID_PPV_ARGS( pCompileResult.GetAddressOf() ) );

    // Error Handling
    ComPtr<IDxcBlobUtf8> pErrors;
    pCompileResult->GetOutput( DXC_OUT_ERRORS, IID_PPV_ARGS( pErrors.GetAddressOf() ), nullptr );
    if ( pErrors && pErrors->GetStringLength() > 0 )
    {
        // Error hadle
        std::cout << (char*)pErrors->GetBufferPointer();
    }

    return pCompileResult;
}