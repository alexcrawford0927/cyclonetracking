@echo off
setlocal enabledelayedexpansion

cd C:\Users\kripa\Documents\project\storm13_200302\height1500

REM ==== Loop through the list of CONTROL file paths ====
FOR /F "usebackq tokens=* delims=" %%n in ("control_names_200302.txt") do (

    REM ==== Clean old files ====
    IF EXIST "CONTROL" DEL "CONTROL"
    IF EXIST "tdump" DEL "tdump"
    IF EXIST "SETUP.CFG" DEL "SETUP.CFG"
    echo Running trajectory for: %%n

    REM ==== Copy current CONTROL file into place ====
    copy /Y "%%n" "CONTROL" >nul

    REM ==== Run HYSPLIT (hyts_std.exe must read from %DIR%) ====
    C:\hysplit\exec\hyts_std.exe
    popd
)

endlocal
pause
