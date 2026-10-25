@rem Topic: start this independent React learning example on Windows.
@rem Input: project folder and installed Node.js; output: a Vite development server.
@rem Flow: locate tools - install missing dependencies - start npm run dev.
@rem No custom functions. npm/Vite perform dependency loading and server startup.
@echo off
chcp 65001 >nul
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo 未找到 Node.js。请先安装 Node.js 22.12 或更新版本，然后重新打开本文件。
  pause
  exit /b 1
)
where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo 未找到 npm。请检查 Node.js 安装及 PATH 设置。
  pause
  exit /b 1
)
if not exist "node_modules\vite\bin\vite.js" (
  echo 正在安装本项目依赖……
  call npm.cmd ci
  if errorlevel 1 (
    echo 依赖安装失败，请查看上面的错误。
    pause
    exit /b 1
  )
)
echo 正在启动示例。结束时在这个窗口按 Ctrl+C。
call npm.cmd run dev -- --open
if errorlevel 1 pause
