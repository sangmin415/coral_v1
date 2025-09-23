# ----------------------------------------------
# Script Recorded by Ansys Electronics Desktop Student Version 2025.2.0
# 13:47:46  Sep 18, 2025
# ----------------------------------------------
import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oDesktop.DeleteProject("Project1")
oProject = oDesktop.NewProject()
oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
oDesktop.DeleteProject("Project1")
oProject = oDesktop.NewProject()
oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
oDesktop.DeleteProject("Project1")
oProject = oDesktop.NewProject()
oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
oDesign = oProject.SetActiveDesign("HFSSDesign1")
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.ImportFromClipboard()
oProject.InsertDesign("HFSS", "HFSSDesign2", "HFSS Terminal Network", "")
oProject.DeleteDesign("HFSSDesign2")
oDesktop.DeleteProject("Project1")
oProject = oDesktop.NewProject()
