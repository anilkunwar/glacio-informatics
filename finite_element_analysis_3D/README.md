
## Mesh Conversion from .grd to Elmer mesh format
$ ElmerGrid 1 2 glacier3D.grd -autoclean

## FEA
The simulation uses a 1D StructuredMeshMapper that can only express variation of Z along X
