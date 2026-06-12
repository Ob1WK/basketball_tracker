@echo off
chcp 65001 >nul 2>&1
title 🏀 Basketball Stats Tracker v2.0

echo.
echo  ██████╗  █████╗ ███████╗██╗  ██╗███████╗████████╗██████╗  █████╗ ██╗     ██╗     
echo  ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║     ██║     
echo  ██████╔╝███████║███████╗█████╔╝ █████╗     ██║   ██████╔╝███████║██║     ██║     
echo  ██╔══██╗██╔══██║╚════██║██╔═██╗ ██╔══╝     ██║   ██╔══██╗██╔══██║██║     ██║     
echo  ██████╔╝██║  ██║███████║██║  ██╗███████╗   ██║   ██████╔╝██║  ██║███████╗███████╗
echo  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝
echo.
echo  Stats Tracker v2.0 - Pickup Basketball Analyzer
echo  ================================================
echo.

:: Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python no encontrado. Instala Python 3.9+ desde python.org
    echo.
    pause
    exit /b 1
)

:: Verificar que existe server.py
if not exist "%~dp0server.py" (
    echo  [ERROR] No se encontro server.py en la carpeta actual.
    echo  Asegurate de que este archivo .bat este en la misma carpeta que server.py
    echo.
    pause
    exit /b 1
)

:: Abrir el browser despues de 2 segundos
echo  Abriendo http://localhost:8000 en tu browser...
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"

echo  Iniciando servidor...
echo.
echo  ┌──────────────────────────────────────────────┐
echo  │  La app se va a abrir en tu browser.         │
echo  │  Para cerrar el servidor, cerra esta ventana │
echo  │  o presiona Ctrl+C                           │
echo  └──────────────────────────────────────────────┘
echo.

:: Arrancar el servidor
python "%~dp0server.py"

:: Si el servidor se cerro, esperar
echo.
echo  Servidor detenido.
pause
