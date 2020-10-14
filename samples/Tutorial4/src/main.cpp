#include <dx12lib/Application.h>

#include <Tutorial4.h>

#define WIN32_LEAN_AND_MEAN
#include <Shlwapi.h>
#include <Windows.h>
#include <shellapi.h>

#include <dxgidebug.h>

using namespace dx12lib;

void ReportLiveObjects()
{
    IDXGIDebug1* dxgiDebug;
    DXGIGetDebugInterface1( 0, IID_PPV_ARGS( &dxgiDebug ) );

    dxgiDebug->ReportLiveObjects( DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_IGNORE_INTERNAL );
    dxgiDebug->Release();
}

int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow )
{
    int retCode = 0;

    WCHAR path[MAX_PATH];

    int     argc = 0;
    LPWSTR* argv = CommandLineToArgvW( lpCmdLine, &argc );
    if ( argv )
    {
        for ( int i = 0; i < argc; ++i )
        {
            // -wd Specify the Working Directory.
            if ( wcscmp( argv[i], L"-wd" ) == 0 )
            {
                wcscpy_s( path, argv[++i] );
                SetCurrentDirectoryW( path );
            }
        }
        LocalFree( argv );
    }

    Application::Create( hInstance );
    {
        std::shared_ptr<Tutorial4> demo =
            std::make_shared<Tutorial4>( L"Learning DirectX 12 - Lesson 4", 1280, 720, true );
        retCode = Application::Get().Run( demo );
    }
    Application::Destroy();

    atexit( &ReportLiveObjects );

    return retCode;
}