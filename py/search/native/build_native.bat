@echo off
setlocal
for %%P in (
  "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
) do (
  if exist %%P (
    call %%P -host_arch=x64 -arch=x64 >nul
    goto :found
  )
)
echo VsDevCmd.bat not found
exit /b 1
:found
cd /d "%~dp0.build"
ninja
exit /b %ERRORLEVEL%
