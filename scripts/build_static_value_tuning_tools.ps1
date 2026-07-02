param(
    [string]$OutputDir = "out\native",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$nativeRoot = Join-Path $repo "py\search\native"
$nativeSrc = Join-Path $nativeRoot "src"
$toolsRoot = Join-Path $repo "tools"
$outDirPath = Join-Path $repo $OutputDir
New-Item -ItemType Directory -Force -Path $outDirPath | Out-Null

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vsDevCmdCandidates = @(
        "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\17\Community\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
    )
    $vsDevCmd = $vsDevCmdCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $vsDevCmd) {
        throw "VsDevCmd.bat not found. Install Visual Studio Build Tools with the MSVC x64 toolchain."
    }

    $tempCmd = [System.IO.Path]::GetTempFileName() + ".cmd"
    try {
        Set-Content -LiteralPath $tempCmd -Encoding ASCII -Value @(
            "@echo off",
            "call `"$vsDevCmd`" -host_arch=x64 -arch=x64 >nul",
            "set"
        )
        $envDump = & cmd.exe /d /c $tempCmd
        foreach ($line in $envDump) {
            $idx = $line.IndexOf("=")
            if ($idx -gt 0) {
                [Environment]::SetEnvironmentVariable($line.Substring(0, $idx), $line.Substring($idx + 1), "Process")
            }
        }
    } finally {
        Remove-Item -LiteralPath $tempCmd -ErrorAction SilentlyContinue
    }
}

$includeFlags = @(
    "/I$nativeRoot",
    "/I$nativeSrc"
)

$commonCompileFlags = @(
    "/nologo",
    "/EHsc",
    "/O2",
    "/Ob3",
    "/Oi",
    "/Ot",
    "/fp:fast",
    "/arch:AVX2",
    "/GL",
    "/DNDEBUG",
    "/DTRIBES_NATIVE_MCTS_STANDALONE",
    "/std:c++17",
    "/bigobj"
) + $includeFlags

$rulesSource = Join-Path $nativeSrc "rules.cpp"
$staticEvalSource = Join-Path $nativeSrc "static_eval.cpp"
$rulesObj = Join-Path $outDirPath "native_rules_tuning.obj"
$staticEvalObj = Join-Path $outDirPath "native_static_eval_tuning.obj"

& cl.exe @commonCompileFlags "/Fo$rulesObj" "/c" $rulesSource
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& cl.exe @commonCompileFlags "/Fo$staticEvalObj" "/c" $staticEvalSource
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$tunerSource = Join-Path $toolsRoot "static_value_result_tuner.cpp"
$tunerExe = Join-Path $outDirPath "static_value_result_tuner.exe"
$tunerObj = Join-Path $outDirPath "static_value_result_tuner.obj"

& cl.exe @commonCompileFlags "/Fo$tunerObj" "/Fe$tunerExe" $tunerSource $rulesObj $staticEvalObj "/link" "/LTCG"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$proxySource = Join-Path $toolsRoot "static_recording_proxy_native.cpp"
$proxyExe = Join-Path $outDirPath "static_recording_proxy_native.exe"
$proxyObj = Join-Path $outDirPath "static_recording_proxy_native.obj"

& cl.exe @commonCompileFlags "/Fo$proxyObj" "/Fe$proxyExe" $proxySource "/link" "/LTCG"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$driverSource = Join-Path $toolsRoot "static_value_result_driver.cpp"
$driverExe = Join-Path $outDirPath "static_value_result_driver.exe"
$driverObj = Join-Path $outDirPath "static_value_result_driver.obj"

& cl.exe @commonCompileFlags "/Fo$driverObj" "/Fe$driverExe" $driverSource "/link" "/LTCG"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Built $tunerExe"
Write-Host "Built $proxyExe"
Write-Host "Built $driverExe"
