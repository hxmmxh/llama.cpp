@echo off
setlocal
set "GGML_VK_DISABLE_COOPMAT=1"
set "LLAMA_DSPARK_DEVICE=1"
set "LLAMA_DSPARK_DENSE_Q=1"
set "LLAMA_DSPARK_ASYNC=1"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" -Mode server %*
exit /b %errorlevel%
