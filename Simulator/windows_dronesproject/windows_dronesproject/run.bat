@echo off
cd /d "%~dp0" 
java -cp "DronesProject.jar;DronesProject_lib/*" drones.Main DroneSimulator 800 600
pause
