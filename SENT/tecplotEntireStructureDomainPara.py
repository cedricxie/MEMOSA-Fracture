import string
from mpi4py import MPI
from numpy import *
import pdb

tectype = {
        'tri' : 'FETRIANGLE',
        'quad' : 'FEQUADRILATERAL',
        'tetra' : 'FETETRAHEDRON',
        'hexa' : 'FEBRICK'
        }

def dumpTecplotEntireStructureDomain(nmesh, meshesLocal, meshesGlobal, mtype, sFields,filename, title_name,Total_count):
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
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [defFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [sXFieldGlobal,MPI.DOUBLE], op=MPI.SUM)    
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [sYFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [sZFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [trXFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [trYFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [trZFieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [J1FieldGlobal,MPI.DOUBLE], op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, [J2FieldGlobal,MPI.DOUBLE], op=MPI.SUM)
  if MPI.COMM_WORLD.Get_rank() == 0:
     file_name = filename
     f = open(file_name, 'a')

     f.write("Title = \" tecplot mesh \" \n")
     f.write("variables = \"x\", \"y\", \"z\", \"DeformationX\", \"DeformationY\", \"DeformationZ\", \"EpsXX\", \"EpsXY\", \"EpsXZ\", \"EpsYY\", \"EpsYZ\", \"EpsZZ\", \"StrXX\", \"StrXY\", \"StrXZ\", \"StrYY\", \"StrYZ\", \"StrZZ\" ,\"J1\", \"J2\"    \n")
     for n in range(0,nmesh):
        ncell  = cellSites[n].getSelfCount()
        nnode  = nodeSites[n].getCount()
        f.write("Zone T = \"%s\" N = %s E = %s DATAPACKING = BLOCK, VARLOCATION = ([4-40]=CELLCENTERED), ZONETYPE=%s\n" %
               (title_name,  nodeSites[n].getCount(), ncell, tectype[mtype]))   
        f.write("StrandID=1, SolutionTime="+str(Total_count)+"\n")             
        #write x
        for i in range(0,nnode):
          f.write(str(coords[n][i][0])+"    ")
	  if ( i % 5 == 4 ):
	     f.write("\n")
        f.write("\n")	  
     
        #write y
        for i in range(0,nnode):
           f.write(str(coords[n][i][1])+"    ")
           if ( i % 5 == 4 ):
	      f.write("\n")
        f.write("\n")	  

        #write z
        for i in range(0,nnode):
           f.write(str(coords[n][i][2])+"    ")
           if ( i % 5 == 4 ):
	      f.write("\n")
        f.write("\n")	 

        #write disp X 
        defFieldGlobal = deformationFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(defFieldGlobal[i][0]) + "    ")
	   if ( i % 5  == 4 ):
	      f.write("\n")
        f.write("\n")
  
        #write disp X 
        defFieldGlobal = deformationFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(defFieldGlobal[i][1]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write disp X 
        defFieldGlobal = deformationFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(defFieldGlobal[i][2]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")
        
        #write StrXX
        sXFieldGlobal = strainXFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(sXFieldGlobal[i][0]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write StrXY
        sXFieldGlobal = strainXFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(sXFieldGlobal[i][1]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write StrXZ
        sXFieldGlobal = strainXFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(sXFieldGlobal[i][2]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write StrYY
        sYFieldGlobal = strainYFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(sYFieldGlobal[i][1]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write StrYZ
        sYFieldGlobal = strainYFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(sYFieldGlobal[i][2]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write StrZZ
        sZFieldGlobal = strainZFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(sZFieldGlobal[i][2]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write StressXX
        trXFieldGlobal = tractXFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(trXFieldGlobal[i][0]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write StressXY
        trXFieldGlobal = tractXFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(trXFieldGlobal[i][1]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write StressXZ
        trXFieldGlobal = tractXFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(trXFieldGlobal[i][2]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write StressYY
        trYFieldGlobal = tractYFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(trYFieldGlobal[i][1]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #write StressYZ
        trYFieldGlobal = tractYFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(trYFieldGlobal[i][2]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")
        
        #write StressZZ
        trZFieldGlobal = tractZFieldGlobal[n]
        for i in range(0,ncell):
           f.write( str(trZFieldGlobal[i][2]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")

        #J1
        J1FieldGlobal = JoneFieldsGlobal[n]
        for i in range(0,ncell):
           f.write( str(J1FieldGlobal[i][0]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")
        
        #J2
        J2FieldGlobal = JtwoFieldsGlobal[n]
        for i in range(0,ncell):
           f.write( str(J2FieldGlobal[i][0]) + "    ")
           if ( i % 5  == 4 ):
              f.write("\n")
        f.write("\n")
         
        #connectivity
        for i in range(0,ncell):
           nnodes_per_cell = cellNodes[n].getCount(i)
           for node in range(0,nnodes_per_cell):
	      f.write( str(cellNodes[n](i,node)+1) + "     ")
           f.write("\n")
	
        f.close()

