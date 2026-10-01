## Undeformed Mesh 
A cuboid of size 1000 x 2000 x 5 m^3

## Surface-bedrock dataset

The distance and elevation profile of glacier and bedrock, matching those of Langtang Himal


## Mesh-deformer

[![continuummodelnt2d](https://img.shields.io/badge/md6-streamlit-green)](https://mesh-deformer6.streamlit.app/ )  (the undeformed mesh is deformed with the information X-Z from the surface-bedrock dataset, the Y-direction is an approximation, advanced r4, a warning system to allow the rational selection of the bedrock and surface .dat files, enables the deformed-hardening in the mesh.nodes so that each mesh is identical, this again will not be compatible with StructuredProjectToPlane solver, Performs the deformation/mesh pre-processing along the Y-axis, for a predefined X-Z information, Computes the elevation angle or slope)
