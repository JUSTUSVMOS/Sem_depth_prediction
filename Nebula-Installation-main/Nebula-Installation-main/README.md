<!-- <div align="center"> -->
<h2>Nebula Simulation Pipeline</h2>

[**GuoWei Wu**]


This is the installation and execution guide for Nebula. Nebula is a software that allows fast simulation of a large number of SEM images using a GPU.



## Installation

Please follow the installation steps on [Nebula Installation](https://nebula-simulator.github.io/).



## Datasets Preparation
Your file structure can be like this.:
```
nebula
|── source
│   │── ...
|── build
│   ├── bin
│   |   │── ...
│   ├── source
│   ├── pmma.mat
│   ├── silicon.mat
│   ├── sem.pri
│   ├── sem-analysis.py
│   ├── sem-analysis_solo_original.py
│   ├── sem-pri.py
│   ├── sem.tri
│   ├── run_nebula.sh
│   ├── triangles_0000.tri
│   ├── ...

```

## Getting Start
Let's go for 🏃‍♀️running code.
### Generate a single SEM image (usually used to check if the generated .tri file has any issues)

Before starting, please try to run it in a Linux environment to avoid potential issues. Also, ensure that you are in the build directory and that all the required packages are properly installed in your environment.
```commandline
export PATH=$PATH:/path/to/your/.../build/bin
python sem-pri.py
nebula_gpu sem.tri sem.pri silicon.mat > output.det
python sem-analysis_solo_original.py output.det
```
### Generate a large number of SEM images at once
```commandline
chmod +x run_nebula.sh
./run_nebula.sh
```
