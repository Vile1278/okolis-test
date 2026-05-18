# OkolisAI
I want to build a system that uses LiDAR scans (iPhone) to reconstruct outdoor environments (yard, house surroundings, walls, terrain), convert them into structured 3D representations, semantically understand them (wall, road, object, ground), and allow the user to interactively edit them (e.g. extend a wall, replace terrain, add structures).


pokretanje:

cd F:\OkolisAI
venv\Scripts\activate
python -m okolis_ai.training.train --config okolis_ai/configs/rtx_a4000.yaml