ATTENTION: README is outdated and will be updated in the future

# ClusterAnalysis_Dash

The ClusterAnalysis_Dash Repo contains a tool to analyze Sat-Problems. This is done by using their instance features and solver times to cluster them into groups.
The goal is to find useful clusters to be able to map good solvers to unkown Sat-instances.

Setup:

  To run the app you need the database files gate.db, base.db and sc2020.db from https://git.scc.kit.edu/fv2117/gbd-data
  Place the database files in a folder and set the environment Variable "DBPATH" to the path of the folder.
  
  The application allows you to export and import your settings as json files. 
  You can define the folder of the exported files in the "JSONPATH" environment variable.
