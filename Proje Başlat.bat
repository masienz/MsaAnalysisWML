@echo off
title ScrappiX
echo.
echo ========================================================
echo   ScrappiX - Sistem Yukleniyor...
echo ========================================================
echo.

:: 1. Ayar dosyasını oluştur (Email sormaması için)
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
(
echo [browser]
echo gatherUsageStats = false
echo [server]
echo headless = false
) > "%USERPROFILE%\.streamlit\config.toml"

:: 2. Uygulamayı çalıştır (Tarayıcıyı kendi açacak)
cd /d "%~dp0"
streamlit run app.py