## Mesh Design for Glacier 3D Morphology

 The Python pre-deformation method of a regular grid  is the industry standard for real glacier simulations because it allows the researchers to use actual GIS/DEM raster data.
 The deformation r1 and r2 are performed on the base glacier3D.grd file. 
 The mesh deformation is performed only on mesh.nodes file , and so the mesh.boundary, mesh.elements and mesh.header files remain unchanged.

[![continuummodelnt2d](https://img.shields.io/badge/md1-streamlit-red)](https://mesh-deformer1.streamlit.app/ )  (Performs the mesh-preprocessing ( deformation ) along the Y-axis, for a predefined X-Z information)

[![continuummodelnt2d](https://img.shields.io/badge/md2-streamlit-red)](https://mesh-deformer2.streamlit.app/ )  (Performs the deformation/mesh pre-processing along the Y-axis, for a predefined X-Z information, Nodes identical e.g. 10303 -1 0 650 0.0000, and 11374 -1 0 650 0.0000, Computes the elevation angle or slope)


[![continuummodelnt2d](https://img.shields.io/badge/md3-streamlit-green)](https://mesh-deformer3.streamlit.app/ )  (Enables the deformed-hardening in the mesh.nodes so that each mesh is identical, this again will not be compatible with StructuredProjectToPlane solver, Performs the deformation/mesh pre-processing along the Y-axis, for a predefined X-Z information, Computes the elevation angle or slope)

