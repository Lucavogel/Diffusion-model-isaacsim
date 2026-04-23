 # Environnement d'exécution (simulation)

 Ce fichier décrit l'environnement minimal nécessaire pour exécuter les simulations et la téléopération (exécution seulement — pas l'entraînement).

 Fichiers disponibles
 - `environment_sim.yml`: environnement conda minimal pour exécution/simulation
 - `environment_repro.yml`: environnement complet (entraînement + exécution)
 - `requirements_sim.txt`: pip packages minimaux pour simulation
 - `requirements.txt`: pip extras pour l'environnement complet

 Installation rapide (conda, recommandée)
 1. Installer Miniconda/Anaconda ou micromamba.
 2. Installer les paquets système (exemple Ubuntu) nécessaires pour l'affichage, MuJoCo et 3D mouse:

 ```bash
 sudo apt update
 sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libspnav-dev spacenavd
 ```

 3. Créer l'environnement minimal pour la simulation:

 ```bash
 conda env create -f environment_sim.yml
 conda activate robodiff_sim
 ```

 Alternative (pip seulement)

 ```bash
 conda create -n robodiff_sim python=3.9 -y
 conda activate robodiff_sim
 pip install -r requirements_sim.txt
 ```

 Notes système importantes
 - MuJoCo (si utilisé) : `free-mujoco-py` requiert les binaires MuJoCo et des paquets système supplémentaires. Voir https://mujoco.org/ et la documentation du repo.
 - `pyrealsense2` (si vous utilisez une RealSense) : nécessite `librealsense`; voir https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
 - `spnav` / 3D mouse : installez `spacenavd` et démarrez le service (`sudo systemctl start spacenavd`) si vous utilisez un 3D mouse.
 - GPU & CUDA : installez des drivers NVIDIA compatibles si vous utilisez GPU. L'environnement `environment_sim.yml` peut inclure `cudatoolkit` pour installation locale mais gardez les drivers système à jour.

 Remarques
 - `environment_sim.yml` est volontairement minimal pour réduire le temps d'installation et les conflits. Si vous avez besoin des dépendances d'entraînement (datasets, diffusers, accelerate, wandb, etc.) utilisez `environment_repro.yml`.
 - Si vous préférez un container Docker ou un `conda-lock` pour reproduire exactement les builds, je peux le générer.

 Contact
 - Dites-moi si vous voulez que je prépare un `environment_sim.yml` spécifique à Ubuntu 22.04 ou un Dockerfile.
