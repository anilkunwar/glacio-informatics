## Convergence Issues

mesh.nodes derived from https://mesh-deformer2.streamlit.app/ , and so the solver faces convergence issues as two or more different nodes tend to get an identical spatial information when the mesh 
is deformed, and their distance is computed as 0.
