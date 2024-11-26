# Main
MainRepo for Compilation

Three Folders:
1) deepSORT
2) SORT
3) shared Modules

Both the deepSORT and SORT models consist of the following modules:
1) Data Association Module
2) Kalman Filter Module
3) Track Management Module
4) Input from a Yolo Module
-----------------------------------------------------
The only difference between these models is the Data Association Module where the deepSORT model uses a CNN for better patttern recognition and data assocation.

The sharedModules Folder will contain the KF, TM, and Yolo Modules for the two models.
The deepSORT and SORT folders will contain their respective main.py and DA Modules.
