1. Must install numba using conda: by run "conda install numba" from terminal.
2. Must install pandas using conda: by run "conda install pandas" or "sudo pip install numba" if you don't have conda, from terminal.
3. Required libraries that I assume are alraedy installed: numpy, opencv, and matplotlib, these can be directly imported.
4. The operations are run from "experiment.py", all functions required are defined in "seam.py", and are imported to "experiment.py".
5. All original images and maps must be placed in a subfolder named "images", all output images will be saved in a subfolder "output".
6. T.csv and S.csv are the transport and step maps saved in csv format, if the script find both these two files in the folder, then it will use them, otherwise it will re-calculate those two maps. Note: this is the most time-consuming step in this whole project.
