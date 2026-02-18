# Digital twin of NICO

## Quickstart
The project is made for Unity 6. Installing this Unity version is a mandatory step, since several features are verifiably 
not functioning in older versions. In order to open the Unity project, clone this repository and import the folder 
`NICO_unity_project` into your Unity hub by clicking `Add project from disk` (Unity 6 should install in this step).

## Scene
After opening the project, in `Assets -> Scenes` is a single scene `nico_scene` containing all the necessary 
components to control and train the NICO model to raise its hands. You can find several scripts attached to the `nico` 
object (located in `nico_setup` game object), which allow two joint manipulation modes - manual (works when `Behavior Parameters`, 
`Nico Agent`, `Decision Requester` are disabled, and `Move With Sliders` is enabled). The joint manipulation using a 
RL policy works when the opposite scripts are used.

## How does it work?
The NICO model is created using a number of GameObjects with meshes and `Articulation Body` 
(https://docs.unity3d.com/Manual/class-ArticulationBody.html) 
components. Game Object `nico` is the root of articulation and is set as immovable. Neck + head and the arms are set 
as articulation bodies, fingers can move too (each has one DoF, except the thumb that has two).

Note: *The real NICO has only 4 DoF in fingers, two for rotating/closing thumb, one for closing index finger, and the last one
for closing middle + pinky fingers. On the other hand, the model can move the middle and pinky fingers separately. If 
you wish to limit the model to move according to the real NICO, it is enough to couple the two fingers via a script.*

The entire model uses simplified objects as colliders, thus during a simulation is able to detect physical interactions. 
For information on how to move the NICO, if you want to modify the scene and/or some settings, best would be to read the 
documentation on articulation bodies, but here are a few tips:
* Check `NicoAgent` script for some useful functions.
* Keep in mind that there are more possible Drive Types to control joints (`Force/Velocity/Target/Target Velocity`), 
use what works best for you. Right now, the model is set to work in `Target` mode.
* Don't forget that the movement is just a physics simulation and setting different physics parameters may have huge
impact on how the movement looks. Don't be afraid to experiment with `Damping`, `Friction`, `Mass` and `Force Limit`.
* Global physics settings (https://docs.unity3d.com/Manual/class-PhysicsManager.html) also have huge impact on the 
behavior. Try modifying `Solver Type`, `Solver Iterations` and `Solver Velocity Iterations`.
* If you want to add more objects into the scene for NICO to interact with, don't forget that each new object needs to
also have a `collider` and a `rigid body` or `articulation body` component. 
  
## Articulation body properties:
This is not necessarily the best-working setup, just a table showing the current values.

| GameObject | Mass | Damping | Joint limits     | Force Limit | Target |
|------------|------|---------|------------------|-------------|--------|
| neck       | 100  | 1       | (-90, 90)    | 100         | 0      |
| head       | 100  | 1       | (-50, 35)    | 100         | 0      |
| shoulder   | 100  | 1       | (-90, 90)    | 1000        | 0      |
| collarbone | 100  | 1       | (-90, 90)    | 100         | 0      |
| upper arm  | 100  | 1       | (-100, 90)   | 1000        | 0      |
| lower arm  | 100  | 1       | (0, 150)     | 500         | 0      |
| forearm    | 100  | 1       | (-90, 150)   | 100         | 0      |
| palm       | 100  | 1       | (-60, 60)    | 100         | 0      |
| thumb1     | 1    | 1e+21   | (-50, 70)    | 100         | 20     |
| thumb2/3/4 | 1    | 1e+21   | (-60, 0)[^2] | 100         | 0      |
| fingers    | 1    | 1e+21   | (0, 60)      | 100         | 0      |

[^2]: The ranges of thumb3/4 are (-15, 0) and their rotation should be 1/4 of the rotation of thumb2. This is due to the
fact, how thumb works in real NICO.

## Training using ML-Agents
If you wish to use ML-agents for Reinforcement Learning (RL), you need to have a conda environment with pytorch and 
ML-agents packages. If the following installation and usage instructions fail to work, for troubleshooting visit 
the [official documentation](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/index.html), 
or the [ML-Agents github repository](https://github.com/Unity-Technologies/ml-agents).

### Environment installation (the order matters)
1. Create conda environment using python 3.10.12 (`conda create -n mlagents python=3.10.12`)
2. Activate the created conda environment (`conda activate mlagents`)
3. Install the latest pytorch version according to the 
   [official PyTorch website](https://pytorch.org/get-started/locally/). 
   If you have GPU in your device, select the "Compute Platform" according to your CUDA version (can be verified for 
   example by typing `nvidia-smi` into your command line).
4. Create a local clone of the ml-agents repository (`clone ml-agents repo: git clone --branch release_23 https://github.com/Unity-Technologies/ml-agents.git`)
5. Go to the cloned directory (`cd /path/to/ml-agents `)
6. Install ml-agents-envs into your conda environment (`python -m pip install ./ml-agents-envs `)
7. Install the ml-agents (`python -m pip install ./ml-agents`)

### Training RL policy
Please, look up the necessary [documentation and examples](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Sample.html) 
for using ml-agents. After correct scene 
setup, activate the installed conda environment in command line. The training is as simple as typing 
`mlagents-learn name_of_the_config_file.yaml --run-id=id_of_this_run`.


### Running on linux servers
There are several things not to forget when training using linux servers.
1) Set up the conda environment exactly as described above, with an exception in step 3: make sure that you have 
   installed torch (version <= 2.8.0) with the correct CUDA version (on uran, neptun & jupiter torch 2.8.0 is tested and
   works with it is CUDA 12.8, 12.9 and 12.9, respectively).
3) You do not need to have a copy of this repository on the server (if possible, avoid it).
4) After setting up the Unity environment on your PC, go to `File -> Build Profiles -> Linux Server`. You might have to 
   install Linux Server in Unity, re-open the project and switch to Linux Server Platform. After doing so, make sure, 
   that in the `Scene List`, only the intended scene is selected. Then build the project into a folder. 
5) In the same folder (which contains the folder with the build), copy your `configuration.yaml` containing training 
   parameters.
6) Copy this folder into the linux server, and you are ready to train. Go to this folder on the server, activate 
   installed conda environment and train (`mlagents-learn name_of_config_file.yaml --env=name_of_the_built_env_folder/file_with.x86_64_but_without.x86_64 --run-id=id_of_this_run`).
7) You might need to change the read, write and execute permissions for the built project 
   (`chmod -R 755 path_to_build/nico_bld.x86_64`). When running the training without having this step, the code might 
   crash with a general broken pipeline error.
   
Notes:
   - This is a headless training – no vision information is used, the simulation runs in the background without 
     graphics.
   - To maximally utilize the server capacities, specify `--num-envs` (10-15 works okay for a simple scene, which 
     already contains 5-15 copies of the agent). Do not be afraid of experiment with these parameters while monitoring 
     the CPU and GPU usage.
   - This headless training can also be tested on your local machine. (for windows: use Windows Server in Unity; also 
     building in windows generates a different hierarchy of files/folders. Thus, make sure you specify the path during 
     training properly).
   - After training, you can move the trained mode and log files to you local machine.
   - While the models are relatively small (as in the number of parameters), the RL training can leverage much more CPU than GPU. Therefore, a clean "CPU only" run can be achieved using "CUDA_VISIBLE_DEVICES=-1 mlagents-learn ..."
