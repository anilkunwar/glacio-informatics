
## Mesh Conversion from .grd to Elmer mesh format
$ ElmerGrid 1 2 glacier3D.grd -autoclean

## FEA
model:
The simulation uses a 1D StructuredMeshMapper that can only express variation of Z along X

modifications - mesh pre-deformation

advanced_model: 
