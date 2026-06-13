param(
    [string]$OutputDir = "out\native",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$nativeRoot = Join-Path $repo "py\search\native"
$nativeSrc = Join-Path $nativeRoot "src"
$source = Join-Path $nativeRoot "bot\native_static_mcts_bot.cpp"
$rulesSource = Join-Path $nativeSrc "native_rules.cpp"
$staticEvalSource = Join-Path $nativeSrc "native_static_eval.cpp"
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

$exe = Join-Path $outDirPath "native_static_mcts_bot.exe"
$obj = Join-Path $outDirPath "native_static_mcts_bot.obj"
$rulesObj = Join-Path $outDirPath "native_rules.obj"
$staticEvalObj = Join-Path $outDirPath "native_static_eval.obj"

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

& cl.exe @commonCompileFlags "/Fo$rulesObj" "/c" $rulesSource
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& cl.exe @commonCompileFlags "/Fo$staticEvalObj" "/c" $staticEvalSource
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$compileFlags = @(
    $commonCompileFlags,
    "/Fo$obj",
    "/Fe$exe",
    $source,
    $rulesObj,
    $staticEvalObj,
    "/link",
    "/LTCG"
)

& cl.exe @compileFlags
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Built $exe"
