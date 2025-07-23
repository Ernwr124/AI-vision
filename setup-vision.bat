@echo off
chcp 65001 > nul
title Установка Terminator Vision

echo.
echo #######################################################
echo #       Установка Terminator Vision - начало          #
echo #######################################################
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo !!! ОШИБКА: Python не установлен или не добавлен в PATH
    echo Пожалуйста, установите Python 3.8+ с сайта python.org
    echo И убедитесь, что выбрали опцию "Add Python to PATH"
    pause
    exit /b 1
)

REM Проверка версии Python
for /f "tokens=2 delims= " %%A in ('python --version 2^>^&1') do set "python_version=%%A"
for /f "tokens=1,2 delims=." %%A in ("%python_version%") do (
    if %%A LSS 3 (
        echo.
        echo !!! ОШИБКА: Требуется Python 3.8 или выше (у вас версия %python_version%)
        pause
        exit /b 1
    )
    if %%A EQU 3 if %%B LSS 8 (
        echo.
        echo !!! ОШИБКА: Требуется Python 3.8 или выше (у вас версия %python_version%)
        pause
        exit /b 1
    )
)

echo.
echo === Установка Python библиотек ===

REM Функция для установки библиотек
:install_lib
set lib_name=%~1
set lib_import=%~2
if "%~2"=="" set lib_import=%lib_name%

echo Проверяем %lib_name%...
python -c "import %lib_import%" >nul 2>&1
if %errorlevel% equ 0 (
    echo %lib_name% уже установлен
    goto :EOF
)

echo Устанавливаем %lib_name%...
pip install %lib_name% --quiet
if %errorlevel% neq 0 (
    echo.
    echo !!! ОШИБКА при установке %lib_name%
    pause
    exit /b 1
)
echo %lib_name% успешно установлен
goto :EOF

REM Установка основных зависимостей
call :install_lib "opencv-python" "cv2"
call :install_lib "mediapipe" "mediapipe"
call :install_lib "numpy" "numpy"

echo.
echo === Проверка веб-камеры ===
python -c "import cv2; print('Камера доступна' if cv2.VideoCapture(0).isOpened() else 'Камера не найдена')"
if %errorlevel% neq 0 (
    echo.
    echo !!! Внимание: возможны проблемы с доступом к камере
)

echo.
echo #######################################################
echo #       Установка завершена!                          #
echo #                                                    #
echo # Для запуска системы выполните:                      #
echo #    python vision.py                                 #
echo #                                                    #
echo # Управление:                                        #
echo #  - Нажмите 'q' для выхода                          #
echo #                                                    #
echo # Требования:                                        #
echo #  - Веб-камера                                      #
echo #  - Хорошее освещение для лучшего распознавания     #
echo #######################################################
echo.

pause