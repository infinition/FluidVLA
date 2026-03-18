@echo off
REM FluidVLA Platform — Windows launcher
REM Run from FluidVLA-main\

setlocal
pushd "%~dp0"

echo.
echo  ███████╗██╗     ██╗   ██╗██╗██████╗ ██╗   ██╗██╗      █████╗
echo  ██╔════╝██║     ██║   ██║██║██╔══██╗██║   ██║██║     ██╔══██╗
echo  █████╗  ██║     ██║   ██║██║██║  ██║██║   ██║██║     ███████║
echo  ██╔══╝  ██║     ██║   ██║██║██║  ██║╚██╗ ██╔╝██║     ██╔══██║
echo  ██║     ███████╗╚██████╔╝██║██████╔╝ ╚████╔╝ ███████╗██║  ██║
echo  ╚═╝     ╚══════╝ ╚═════╝ ╚═╝╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝
echo.
echo  Platform starting on http://localhost:7860
echo  Press Ctrl+C to stop
echo.

:run_with_env_python
if defined FLUIDVLA_PYTHON if exist "%FLUIDVLA_PYTHON%" (
	echo Using FLUIDVLA_PYTHON=%FLUIDVLA_PYTHON%
	"%FLUIDVLA_PYTHON%" fluidvla_server.py --port 7860 --host 0.0.0.0
	goto end
)

if defined CONDA_PYTHON_EXE if exist "%CONDA_PYTHON_EXE%" (
	echo Using CONDA_PYTHON_EXE=%CONDA_PYTHON_EXE%
	"%CONDA_PYTHON_EXE%" fluidvla_server.py --port 7860 --host 0.0.0.0
	goto end
)

where python >nul 2>nul
if not errorlevel 1 (
	echo Using python from PATH
	python fluidvla_server.py --port 7860 --host 0.0.0.0
	goto end
)

where py >nul 2>nul
if not errorlevel 1 (
	echo Using py -3 launcher
	py -3 fluidvla_server.py --port 7860 --host 0.0.0.0
	goto end
)

echo No Python interpreter found.
echo Set FLUIDVLA_PYTHON to a valid python.exe, activate your environment, or install Python on PATH.

:end
popd
pause
