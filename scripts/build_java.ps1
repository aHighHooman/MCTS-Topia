$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$gameRoot = if ($env:TRIBES_GAME_ROOT) {
    Resolve-Path $env:TRIBES_GAME_ROOT
} else {
    Resolve-Path "C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes"
}
$gameSrc = Join-Path $gameRoot "src"
$jsonJar = Join-Path $gameRoot "lib\json.jar"
if (-not (Test-Path -LiteralPath $jsonJar)) {
    $jsonJar = Join-Path $repoRoot "lib\json.jar"
}

if (Test-Path -LiteralPath (Join-Path $repoRoot "out")) {
    Remove-Item -LiteralPath (Join-Path $repoRoot "out") -Recurse -Force
}
New-Item -ItemType Directory -Force (Join-Path $repoRoot "out") | Out-Null

$sources = Get-ChildItem -LiteralPath $gameSrc -Recurse -Filter *.java | ForEach-Object { $_.FullName }
& "$env:JAVA_HOME\bin\javac.exe" -cp $jsonJar -d (Join-Path $repoRoot "out") $sources

$terrainProbs = Join-Path $gameRoot "terrainProbs.json"
if (Test-Path -LiteralPath $terrainProbs) {
    Copy-Item -LiteralPath $terrainProbs -Destination (Join-Path $repoRoot "terrainProbs.json") -Force
}
