param(
    [string]$Python = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildTree = Join-Path $Root "build-native"
$Output = Join-Path $Root "build"

if (-not $Python) {
    $preferred = "python"
    if (Test-Path -LiteralPath $preferred) {
        $Python = $preferred
    } else {
        $Python = (Get-Command python -ErrorAction Stop).Source
    }
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}
if ($Clean -and (Test-Path -LiteralPath $BuildTree)) {
    Remove-Item -LiteralPath $BuildTree -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $BuildTree, $Output | Out-Null

$Gpp = Get-Command g++ -ErrorAction SilentlyContinue
if ($Gpp) {
    Write-Host "[1/3] Build JunqiAZ directly with g++"
    $PybindInclude = (& $Python -c "import pybind11; print(pybind11.get_include())").Trim()
    $PythonInclude = (& $Python -c "import sysconfig; print(sysconfig.get_paths()['include'])").Trim()
    $PythonLib = (& $Python -c "import sys; print(sys.base_prefix + r'\libs')").Trim()
    $PythonTag = (& $Python -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}')").Trim()
    $Extension = (& $Python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))").Trim()
    $Sources = @(
        "junqi_board.cpp", "junqi_state.cpp", "setup_validator.cpp",
        "move_generator.cpp", "combat_resolver.cpp", "setup_generator.cpp",
        "junqi_environment.cpp", "az_state.cpp", "neural_puct.cpp",
        "heuristic_084_evaluator.cpp",
        "junqi_az_bindings.cpp"
    ) | ForEach-Object { Join-Path $Root "cpp\$_" }
    $Target = Join-Path $Output "JunqiAZ$Extension"
    $CompileArgs = @(
        "-O3", "-DNDEBUG", "-std=c++17", "-shared", "-static",
        "-I$PybindInclude", "-I$PythonInclude"
    ) + $Sources + @("-L$PythonLib", "-l$PythonTag", "-o", $Target)
    & $Gpp.Source @CompileArgs
    if ($LASTEXITCODE -ne 0) { throw "Direct C++ build failed." }
    Write-Host "[2/3] Build independent JunqiOracle adapter"
    $OracleSources = @(
        (Join-Path $Root "supplement\src\root_only_runtime.cpp"),
        (Join-Path $Root "cpp\oracle_bindings.cpp")
    )
    $OracleTarget = Join-Path $Output "JunqiOracle$Extension"
    $OracleArgs = @(
        "-O3", "-DNDEBUG", "-std=c++17", "-shared", "-static",
        "-I$PybindInclude", "-I$PythonInclude",
        "-I$(Join-Path $Root 'supplement\src')"
    ) + $OracleSources + @("-L$PythonLib", "-l$PythonTag", "-o", $OracleTarget)
    & $Gpp.Source @OracleArgs
    if ($LASTEXITCODE -ne 0) { throw "JunqiOracle build failed." }
} else {
    Write-Host "[1/3] Configure JunqiAZ with CMake"
    $Configure = @(
        "-S", (Join-Path $Root "cpp"), "-B", $BuildTree,
        "-DCMAKE_BUILD_TYPE=Release", "-DPython_EXECUTABLE=$Python",
        "-DJUNQI_AZ_OUTPUT_DIR=$Output"
    )
    if (Get-Command ninja -ErrorAction SilentlyContinue) {
        $Configure += @("-G", "Ninja")
    }
    & cmake @Configure
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
    Write-Host "[2/3] Build JunqiAZ"
    & cmake --build $BuildTree --config Release --parallel
    if ($LASTEXITCODE -ne 0) { throw "C++ build failed." }
}

Write-Host "[3/3] Native smoke test"
$env:PYTHONPATH = "$Output;$(Join-Path $Root 'python');$env:PYTHONPATH"
& $Python (Join-Path $Root "cpp\smoke_test.py")
if ($LASTEXITCODE -ne 0) { throw "Native smoke test failed." }

Write-Host "JunqiAZ build is ready: $Output"
