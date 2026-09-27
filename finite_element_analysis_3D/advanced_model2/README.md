# The solver run converges perfectly 

The deformation hardening methodology applied in https://mesh-deformer3.streamlit.app/ (md3 app), ensures the uniqueness of each nodes even in the aftermath of bedrock deformation.
For this model, the U-Valley (Parabolic) profile is chosen along Y-direction.

# Limitation

Such pre-processing renders the model incompatible with the procedure "StructuredProjectToPlane" used in the postprocessing of the solutions to measure height and depth. 
So, Solver 1 (Equation = HeightDepth) cannot be utilized when using with the mesh generated via the md3 app
