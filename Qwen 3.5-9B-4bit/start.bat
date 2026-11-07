@echo off
setlocal enabledelayedexpansion
title Gideon - Qwen3.5-9B
cd /d "%~dp0"

echo ===================================================
echo   AVVIO SERVER LOCAL LLM - QWEN3.5-9B (GGUF Q4_K_M)
echo ===================================================

:: Impostazioni opzionali (togli i due punti iniziali per attivarle)
:: set CTX=16384
:: set ENABLE_THINKING=0
:: set MAX_OCR_PAGES=20
:: set BRAVE_API_KEY=incolla-qui-la-tua-chiave
:: set ENABLE_WEB=0
:: set SEARXNG_URL=http://localhost:8080
:: set WEB_REGION=it-it
:: set WEB_DEEP_RESULTS=2
:: set LLAMA_EXE=C:\percorso\llama-server.exe

:: =====================================================
:: 1. VERIFICA O CREAZIONE DELL'AMBIENTE PYTHON (shared_venv)
:: =====================================================
set VENV_DIR=%~dp0..\shared_venv
set PY=%VENV_DIR%\Scripts\python.exe

if not exist "%PY%" (
    echo.
    echo [1/4] Ambiente virtuale non trovato in: "%VENV_DIR%"
    echo Configurazione automatica dell'ambiente Python...
    
    set "SYS_PY="
    python --version >nul 2>&1
    if not errorlevel 1 (
        set "SYS_PY=python"
    ) else (
        py -3 --version >nul 2>&1
        if not errorlevel 1 (
            set "SYS_PY=py -3"
        )
    )
    
    if not defined SYS_PY (
        echo.
        echo ERRORE CRITICO: Python non e' installato o non e' presente nel PATH.
        echo Per avviare Gideon e' necessario installare Python [consigliato 3.11+].
        echo Scaricalo da: https://www.python.org/downloads/
        echo Durante l'installazione ricordati di spuntare "Add python.exe to PATH".
        echo.
        pause
        exit /b 1
    )
    
    echo Utilizzo Python di sistema: !SYS_PY!
    echo Creazione del virtual environment 'shared_venv'...
    !SYS_PY! -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo ERRORE: Creazione dell'ambiente virtuale fallita.
        pause
        exit /b 1
    )
    echo Ambiente virtuale creato con successo.
)

:: =====================================================
:: 2. VERIFICA E INSTALLAZIONE DELLE DIPENDENZE PYTHON
:: =====================================================
"%PY%" -c "import fastapi, uvicorn, httpx, pymupdf, ddgs, trafilatura, huggingface_hub" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [2/4] Installazione delle dipendenze Python mancanti...
    echo Pacchetti: fastapi uvicorn httpx pymupdf ddgs trafilatura huggingface_hub
    "%PY%" -m pip install --upgrade pip
    if exist "%~dp0..\requirements.txt" (
        "%PY%" -m pip install -r "%~dp0..\requirements.txt"
    ) else (
        "%PY%" -m pip install fastapi uvicorn httpx pymupdf ddgs trafilatura huggingface_hub
    )
    if errorlevel 1 (
        echo.
        echo ATTENZIONE: Si e' verificato un problema nell'installazione dei pacchetti Python.
        pause
    ) else (
        echo Dipendenze Python pronte.
    )
)

:: =====================================================
:: 3. VERIFICA O DOWNLOAD DEI MODELLI (shared_models)
:: =====================================================
set MODEL_DIR=%~dp0..\shared_models\qwen3.5-9b
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

:: Verifica modello linguistico principale (*.gguf escluso mmproj)
"%PY%" -c "from pathlib import Path; d=Path(r'%MODEL_DIR%'); gg=list(d.rglob('*.gguf')); exit(0 if any(not f.name.lower().startswith('mmproj') for f in gg) else 1)"
if errorlevel 1 (
    echo.
    echo [3/4] Modello linguistico Qwen3.5-9B non trovato in:
    echo       "%MODEL_DIR%"
    echo Avvio il download di Qwen3.5-9B-Q4_K_M.gguf [circa 5.6 GB] da Hugging Face...
    echo [Questo passaggio puo' richiedere qualche minuto in base alla tua connessione]
    "%PY%" -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3.5-9B-GGUF', 'Qwen3.5-9B-Q4_K_M.gguf', local_dir=r'%MODEL_DIR%')"
    if errorlevel 1 (
        echo.
        echo ERRORE: Download del modello linguistico non riuscito.
        echo Verifica la connessione a Internet o scarica manualmente il file da:
        echo https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
        echo e salvalo in: "%MODEL_DIR%"
        pause
        exit /b 1
    )
    echo Modello linguistico scaricato con successo.
)

:: Verifica proiettore visivo multimodale (mmproj)
"%PY%" -c "from pathlib import Path; d=Path(r'%MODEL_DIR%'); gg=list(d.rglob('*.gguf')); exit(0 if any(f.name.lower().startswith('mmproj') for f in gg) else 1)"
if errorlevel 1 (
    echo.
    echo [3/4] Proiettore visivo mmproj non trovato in:
    echo       "%MODEL_DIR%"
    echo Avvio il download di mmproj-F16.gguf [circa 900 MB] per supporto immagini e OCR PDF...
    "%PY%" -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3.5-9B-GGUF', 'mmproj-F16.gguf', local_dir=r'%MODEL_DIR%')"
    if errorlevel 1 (
        echo ATTENZIONE: Download di mmproj non riuscito. Il server funzionera' solo in modalita' testo.
    ) else (
        echo Proiettore visivo scaricato con successo.
    )
)

:: =====================================================
:: 4. VERIFICA DI LLAMA-SERVER (llama.cpp)
:: =====================================================
if not defined LLAMA_EXE (
    where llama-server >nul 2>&1
    if errorlevel 1 (
        rem Cerca nelle directory tipiche di installazione di WinGet
        set "FOUND_LLAMA="
        for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp*") do (
            if exist "%%D\llama-server.exe" set "FOUND_LLAMA=%%D\llama-server.exe"
        )
        if defined FOUND_LLAMA (
            set "LLAMA_EXE=!FOUND_LLAMA!"
        ) else (
            echo.
            echo [4/4] 'llama-server' non trovato nel sistema.
            where winget >nul 2>&1
            if not errorlevel 1 (
                echo Tento l'installazione automatica di llama.cpp tramite winget...
                winget install --id ggml.llamacpp -e --accept-source-agreements --accept-package-agreements
                for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp*") do (
                    if exist "%%D\llama-server.exe" set "FOUND_LLAMA=%%D\llama-server.exe"
                )
                if defined FOUND_LLAMA (
                    set "LLAMA_EXE=!FOUND_LLAMA!"
                    echo llama-server installato con successo: !LLAMA_EXE!
                )
            )
            if not defined LLAMA_EXE (
                echo.
                echo ATTENZIONE: llama-server non e' stato trovato ne' installato automaticamente.
                echo Se all'avvio ricevi un errore su llama-server, installalo con:
                echo    winget install ggml.llamacpp
                echo oppure scaricalo da: https://github.com/ggml-org/llama.cpp/releases
                echo.
            )
        )
    )
)

:: =====================================================
:: 5. AVVIO DELL'APPLICAZIONE GIDEON
:: =====================================================
:: Trova l'IP locale della macchina (primo IPv4 trovato)
set IP=
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    if not defined IP set IP=%%a
)
if defined IP set IP=%IP: =%
if not defined IP set IP=IP-DEL-PC

echo.
echo ===================================================
echo Il modello si sta caricando sulla GPU (puo' richiedere un minuto).
echo Quando compare "Application startup complete" puoi collegarti da:
echo.
echo    * http://localhost:8000      (da questo PC)
echo    * http://%IP%:8000      (da altri PC/smartphone nella stessa rete)
echo.
echo Per chiudere: premi Ctrl+C in questa finestra.
echo ===================================================

:: server.py avvia e ferma da solo llama-server (log in llama-server.log)
"%PY%" server.py

pause