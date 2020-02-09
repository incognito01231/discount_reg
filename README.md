

Code for the ICML anonymous submission "Discount Factor as a Regularizer in Reinforcement Learning"

The code is based on the [Ray and RLlib](https://github.com/ray-project/ray) frameworks.

# Prerequisites 
* Anaconda Python 3.7

Only for Mujoco experiments:
* Linux\MacOS
* Nvidia GPU
* CUDA + CudNN
* [Mujoco-Pro version 150](http://mujoco.org/) 


# Installation guidelines
* Create an Anaconda Python 3.7 environment
* conda install numpy matplotlib scipy
*  pip install lz4 psutil pandas glfw setproctitle lockfile 
*  pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.9.0.dev0-cp37-cp37m-manylinux1_x86_64.whl
* Go to <project_dir>  and run $ python3 setup.py install


Only for Mujoco experiments:
* Install TensorFlow GPU version 2.0
* Install  [mujoco-py - version 1.50.1.0](https://github.com/openai/mujoco-py/edit/master/README.md) and [Gym version 0.15.3](https://github.com/openai/gym)
 * Go to <project_dir>/python/ray  and run setup-dev.py, answer y to all options



#  Recreating tabular experiments
* Chain experiment: run main_MRP.py 
* Single regularization method: run main_control.py
* Varying k in RandomMDP: run_k_grid.py
* 2 regularization methods:  run run_2d_reg_grid.py.py


#  Recreating Mujoco experiments 
* Run main_ray.py