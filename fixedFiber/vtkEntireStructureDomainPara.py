
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

def dumpvtkEntireStructureDomain(geomFields, nmesh, meshesLocal, meshesGlobal, mtype, sFields, file_name, title_name,nstep):
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

  deformationFields = []
  for n in range(0,nmesh):
     deformationFields.append( sFields.deformation[cellSitesLocal[n]].asNumPyArray() )    


  strainXFields = []
  for n in range(0,nmesh):
     strainXFields.append( sFields.strainX[cellSitesLocal[n]].asNumPyArray() )

  strainYFields = []
  for n in range(0,nmesh):
     strainYFields.append( sFields.strainY[cellSitesLocal[n]].asNumPyArray() )

  strainZFields = []
  for n in range(0,nmesh):
     strainZFields.append( sFields.strainZ[cellSitesLocal[n]].asNumPyArray() )

  tractXFields = []
  for n in range(0,nmesh):
     tractXFields.append( sFields.tractionX[cellSitesLocal[n]].asNumPyArray() )

  tractYFields = []
  for n in range(0,nmesh):
     tractYFields.append( sFields.tractionY[cellSitesLocal[n]].asNumPyArray() )

  tractZFields = []
  for n in range(0,nmesh):
     tractZFields.append( sFields.tractionZ[cellSitesLocal[n]].asNumPyArray() )



  #opening global Array 
  deformationFieldGlobal = []
  strainXFieldGlobal = []
  strainYFieldGlobal = []
  strainZFieldGlobal = []
  tractXFieldGlobal = []
  tractYFieldGlobal = []
  tractZFieldGlobal = []
  JoneFieldsGlobal = []
  JtwoFieldsGlobal = []
  for n in range(0,nmesh):

    selfCount = cellSites[n].getSelfCount()
    #Initiate Global Field"W/ S"
    deformationFieldGlobal.append( zeros((selfCount,3), float) ) #if it is velocoity (selfCount,3) 
    strainXFieldGlobal.append( zeros((selfCount,3), float) ) #if it is velocity (selfCount,3)
    strainYFieldGlobal.append( zeros((selfCount,3), float) ) #if it is velocity (selfCount,3)
    strainZFieldGlobal.append( zeros((selfCount,3), float) ) #if it is velocity (selfCount,3)
    tractXFieldGlobal.append( zeros((selfCount,3), float) ) #if it is velocity (selfCount,3)
    tractYFieldGlobal.append( zeros((selfCount,3), float) ) #if it is velocity (selfCount,3)
    tractZFieldGlobal.append( zeros((selfCount,3), float) ) #if it is velocity (selfCount,3)
    #J1 and J2 Field
    JoneFieldsGlobal.append(zeros((selfCount,1), float))
    JtwoFieldsGlobal.append(zeros((selfCount,1), float))
    meshLocal = meshesLocal[n]
    localToGlobal = meshLocal.getLocalToGlobalPtr().asNumPyArray()
    #Initiate Global Field"W/O S"
    defFieldGlobal  = deformationFieldGlobal[n]
    sXFieldGlobal  = strainXFieldGlobal[n]  
    sYFieldGlobal  = strainYFieldGlobal[n]
    sZFieldGlobal  = strainZFieldGlobal[n]
    trXFieldGlobal  = tractXFieldGlobal[n]
    trYFieldGlobal  = tractYFieldGlobal[n]
    trZFieldGlobal  = tractZFieldGlobal[n]
    defFieldLocal   = deformationFields[n]
    #J1 and J2 Field
    J1FieldGlobal = JoneFieldsGlobal[n]
    J2FieldGlobal = JtwoFieldsGlobal[n]
    #Initiate Local Field"W/O S"
    sXFieldLocal   = strainXFields[n]
    sYFieldLocal   = strainYFields[n]
    sZFieldLocal   = strainZFields[n]
    trXFieldLocal   = tractXFields[n]
    trYFieldLocal   = tractYFields[n]
    trZFieldLocal   = tractZFields[n]
    #fill local part of cells
    selfCount = cellSitesLocal[n].getSelfCount()
    
    
    cellNodeCount = array([0.0])
    ncell  = cellSites[n].getSelfCount()
    nnode  = nodeSites[n].getCount()
    
    for i in range(0,selfCount):
       globalID               = localToGlobal[i]
       defFieldGlobal[globalID] = defFieldLocal[i]
       sXFieldGlobal[globalID] = sXFieldLocal[i]
       sYFieldGlobal[globalID] = sYFieldLocal[i]
       sZFieldGlobal[globalID] = sZFieldLocal[i]
       trXFieldGlobal[globalID] = trXFieldLocal[i]
       trYFieldGlobal[globalID] = trYFieldLocal[i]
       trZFieldGlobal[globalID] = trZFieldLocal[i]
       #J1 and J2 Field
       J1FieldGlobal[globalID] = sXFieldLocal[i][0] + sYFieldLocal[i][1] + sZFieldLocal[i][2]
       J2FieldGlobal[globalID] = 0.5*((sXFieldLocal[i][0]-sYFieldLocal[i][1])**2+\
       (sYFieldLocal[i][1]-sZFieldLocal[i][2])**2+(sZFieldLocal[i][2]-sXFieldLocal[i][0])**2+\
       6.0*(sXFieldLocal[i][1]**2+sXFieldLocal[i][2]**2+sYFieldLocal[i][2]**2))
       
       cellNodeCount[0] += cellNodes[n].getCount(i)+1
       
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [defFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [sXFieldGlobal,MPI.DOUBLE], op=MPI.SUM)    
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [sYFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [sZFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [trXFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [trYFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [trZFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [J1FieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [J2FieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [cellNodeCount,MPI.DOUBLE], op=MPI.SUM)

  if MPI.COMM_WORLD.Get_rank() == 0:
  
     file_name_vtk = "structure-"+str(nstep)+".vtk"
     
     f = open(file_name_vtk, 'w')
     
     f.write("# vtk DataFile Version 2.0\n")
     f.write("structure output\n")
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
     
     f.write("VECTORS "+"Deformation"+" double\n")
     defFieldGlobal = deformationFieldGlobal[n]
     for i in range(0,ncell):
         f.write(str(defFieldGlobal[i][0])+" "+str(defFieldGlobal[i][1])+" "+str(defFieldGlobal[i][2])+"\n")
     
     f.write("VECTORS "+"StrX"+" double\n")
     sXFieldGlobal = strainXFieldGlobal[n]
     for i in range(0,ncell):
         f.write(str(sXFieldGlobal[i][0])+" "+str(sXFieldGlobal[i][1])+" "+str(sXFieldGlobal[i][2])+"\n")
          
     f.write("VECTORS "+"StrY"+" double\n")
     sYFieldGlobal = strainYFieldGlobal[n]
     for i in range(0,ncell):
         f.write(str(sYFieldGlobal[i][0])+" "+str(sYFieldGlobal[i][1])+" "+str(sYFieldGlobal[i][2])+"\n")
 
     f.write("VECTORS "+"StrZ"+" double\n")
     sZFieldGlobal = strainZFieldGlobal[n]
     for i in range(0,ncell):
         f.write(str(sZFieldGlobal[i][0])+" "+str(sZFieldGlobal[i][1])+" "+str(sZFieldGlobal[i][2])+"\n")

     f.write("VECTORS "+"TractX"+" double\n")
     trXFieldGlobal = tractXFieldGlobal[n]
     for i in range(0,ncell):
         f.write(str(trXFieldGlobal[i][0])+" "+str(trXFieldGlobal[i][1])+" "+str(trXFieldGlobal[i][2])+"\n")
          
     f.write("VECTORS "+"TractY"+" double\n")
     trYFieldGlobal = tractYFieldGlobal[n]
     for i in range(0,ncell):
         f.write(str(trYFieldGlobal[i][0])+" "+str(trYFieldGlobal[i][1])+" "+str(trYFieldGlobal[i][2])+"\n")
 
     f.write("VECTORS "+"TractZ"+" double\n")
     trZFieldGlobal = tractZFieldGlobal[n]
     for i in range(0,ncell):
         f.write(str(trZFieldGlobal[i][0])+" "+str(trZFieldGlobal[i][1])+" "+str(trZFieldGlobal[i][2])+"\n")

     f.close()
