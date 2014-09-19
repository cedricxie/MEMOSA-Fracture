
import string
from mpi4py import MPI
from numpy import *
import pdb
import fvm.exporters_atyped_double as exporters

tectype = {
        'tri' : 'FETRIANGLE',
        'quad' : 'FEQUADRILATERAL',
        'tetra' : 'FETETRAHEDRON',
        'hexa' : 'FEBRICK'
        }

VTK_EMPTY_CELL        =0
VTK_VERTEX            =1
VTK_POLY_VERTEX       =2
VTK_LINE              =3
VTK_POLY_LINE         =4
VTK_TRIANGLE          =5
VTK_TRIANGLE_STRIP    =6
VTK_POLYGON           =7
VTK_PIXEL             =8
VTK_QUAD              =9
VTK_TETRA             =10
VTK_VOXEL             =11
VTK_HEXAHEDRON        =12
VTK_WEDGE             =13
VTK_PYRAMID           =14
VTK_PENTAGONAL_PRISM  =15
VTK_HEXAGONAL_PRISM   =16
VTK_CONVEX_POINT_SET  =41

dim=3

def dumpvtkEntireFractureDomain(geomFields, nmesh, meshesLocal, meshesGlobal, mtype, tFields, sFields, file_name, title_name,nstep):

  #cell sites
  cellSites = []
  for n in range(0,nmesh):
     cellSites.append( meshesGlobal[n].getCells() )

  #face sites
  faceSites = []
  for n in range(0,nmesh):
     faceSites.append( meshesGlobal[n].getFaces() )

  #node sites
  nodeSites = []
  for n in range(0,nmesh):
     nodeSites.append( meshesGlobal[n].getNodes() )

  #get connectivity (faceCells)
  faceCells = []
  for n in range(0,nmesh):
     faceCells.append( meshesGlobal[n].getConnectivity( faceSites[n], cellSites[n] ) )
 
  #get connectivity ( cellNodes )
  cellNodes = []
  for n in range(0,nmesh):
     cellNodes.append( meshesGlobal[n].getCellNodes() )

  #coords
  coords = []
  for n in range(0,nmesh):
     coords.append( meshesGlobal[n].getNodeCoordinates().asNumPyArray() )
 
  cellSitesLocal = []
  for n in range(0,nmesh):
     cellSitesLocal.append( meshesLocal[n].getCells() )

  fractureFields = []
  for n in range(0,nmesh):
     fractureFields.append( tFields.phasefieldvalue[cellSitesLocal[n]].asNumPyArray() )    

  fracturegradFields = []
  for n in range(0,nmesh):
     fracturegradFields.append( tFields.phasefieldGradient[cellSitesLocal[n]].asNumPyArray() )
     
  deformationFields = []
  for n in range(0,nmesh):
     deformationFields.append( sFields.deformation[cellSitesLocal[n]].asNumPyArray() )  

  #opening global Array 
  deformationFieldGlobal = []
  fractureFieldGlobal = []
  fracturegradFieldGlobal = []

  for n in range(0,nmesh):

    selfCount = cellSites[n].getSelfCount()
    deformationFieldGlobal.append( zeros((selfCount,3), float) ) #if it is velocoity (selfCount,3) 
    fractureFieldGlobal.append( zeros((selfCount,1), float) ) #if it is velocoity (selfCount,3) 
    fracturegradFieldGlobal.append( zeros((selfCount,3), float) ) #if it is velocity (selfCount,3)
    meshLocal = meshesLocal[n]
    localToGlobal = meshLocal.getLocalToGlobalPtr().asNumPyArray()
    defFieldGlobal  = deformationFieldGlobal[n]
    tFieldGlobal  = fractureFieldGlobal[n]
    tgradFieldGlobal  = fracturegradFieldGlobal[n]  
    defFieldLocal   = deformationFields[n]
    tFieldLocal   = fractureFields[n]
    tgradFieldLocal   = fracturegradFields[n]
    #fill local part of cells
    selfCount = cellSitesLocal[n].getSelfCount()
    
    
    cellNodeCount = array([0.0])
    ncell  = cellSites[n].getSelfCount()
    nnode  = nodeSites[n].getCount()
    
    for i in range(0,selfCount):
       globalID               = localToGlobal[i]
       defFieldGlobal[globalID] = defFieldLocal[i]
       tFieldGlobal[globalID] = tFieldLocal[i]
       tgradFieldGlobal[globalID] = tgradFieldLocal[i]
       cellNodeCount[0] += cellNodes[n].getCount(i)+1
             
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [tFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [tgradFieldGlobal,MPI.DOUBLE], op=MPI.SUM)    
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [defFieldGlobal,MPI.DOUBLE], op=MPI.SUM) 
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [cellNodeCount,MPI.DOUBLE], op=MPI.SUM)

  if MPI.COMM_WORLD.Get_rank() == 0:
  
     file_name_vtk = "fracture-"+str(nstep)+".vtk"
     
     f = open(file_name_vtk, 'w')
     
     f.write("# vtk DataFile Version 2.0\n")
     f.write("fracture output\n")
     f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
     f.write("POINTS "+str(nnode)+" double\n")
     
     for i in range(0,nnode):
         f.write(str(coords[n][i][0])+ " " + str(coords[n][i][1])+ " " +str(coords[n][i][2])+"\n")
     
     f.write("CELLS "+str(ncell)+ " " + str(int(cellNodeCount[0]))+"\n")
     
     for i in range(0,ncell):
         f.write(str(cellNodes[n].getCount(i))+" ")
         for j in range(0,cellNodes[n].getCount(i)):
             f.write(str(cellNodes[n](i,j))+" ")
         f.write("\n")

     f.write("CELL_TYPES "+ str(ncell) +"\n")
     
     for i in range(0,ncell):
         vtkCellType = VTK_CONVEX_POINT_SET
         nCellNodes=cellNodes[n].getCount(i)
         if dim == 2:
             if nCellNodes == 4:
                 vtkCellType = VTK_QUAD
             elif nCellNodes == 3:
                 vtkCellType = VTK_TRIANGLE
             else:
                 vtkCellType = VTK_POLYGON
         else:
             if nCellNodes == 4:
                 vtkCellType = VTK_TETRA
             elif nCellNodes == 8:
                 vtkCellType = VTK_HEXAHEDRON
             elif nCellNodes == 5:
                 vtkCellType = VTK_PYRAMID
             elif nCellNodes == 6:
                 vtkCellType = VTK_WEDGE
         f.write(str(vtkCellType)+"\n")
 
     f.write("CELL_DATA "+str(ncell)+"\n")
     
     f.write("SCALARS "+"PhaseField"+" double\n")
     f.write("LOOKUP_TABLE default\n")
     for i in range(0,ncell):
         f.write(str(tFieldGlobal[i][0])+"\n") 
 
     f.write("VECTORS "+"Deformation"+" double\n")
     defFieldGlobal = deformationFieldGlobal[n]
     for i in range(0,ncell):
         f.write(str(defFieldGlobal[i][0])+" "+str(defFieldGlobal[i][1])+" "+str(defFieldGlobal[i][2])+"\n")
     f.close()