#!/usr/bin/env python

#--------------------------------------------------------------------------------------
## pythonFlu - Python wrapping for OpenFOAM C++ API
## Copyright (C) 2010- Alexey Petrov
## Copyright (C) 2009-2010 Pebble Bed Modular Reactor (Pty) Limited (PBMR)
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
## 
## See http://sourceforge.net/projects/pythonflu
##
## Author : Alexey PETROV
##


#----------------------------------------------------------------------------
from Foam import ref, man


#----------------------------------------------------------------------------
def readTransportProperties( runTime, mesh ):
    
    ref.ext_Info() << "Reading transportProperties\n" << ref.nl
    transportProperties = man.IOdictionary( man.IOobject( ref.word( "transportProperties" ),
                                                          ref.fileName( runTime.constant() ),
                                                           mesh,
                                                          ref.IOobject.MUST_READ_IF_MODIFIED,
                                                          ref.IOobject.NO_WRITE ) )

    nu = ref.dimensionedScalar( transportProperties.lookup( ref.word( "nu" ) ) )

    return transportProperties, nu


#----------------------------------------------------------------------------
def _createFields( runTime, mesh ):
        
    ref.ext_Info() << "Reading field p\n" << ref.nl
    p = man.volScalarField( man.IOobject( ref.word( "p" ),
                                          ref.fileName( runTime.timeName() ),
                                          mesh,
                                          ref.IOobject.MUST_READ,
                                          ref.IOobject.AUTO_WRITE ),
                            mesh )

    ref.ext_Info() << "Reading field U\n" << ref.nl
    U = man.volVectorField( man.IOobject( ref.word( "U" ),
                                          ref.fileName( runTime.timeName() ),
                                          mesh,
                                          ref.IOobject.MUST_READ,
                                          ref.IOobject.AUTO_WRITE ),
                            mesh )

    phi = man.createPhi( runTime, mesh, U )

    return p, U, phi
    

#--------------------------------------------------------------------------------------
def readTurbulenceProperties( runTime, mesh, U ):
    
    ref.ext_Info() << "Reading turbulenceProperties\n" << ref.nl
    turbulenceProperties = man.IOdictionary( man.IOobject( ref.word( "turbulenceProperties" ),
                                                           ref.fileName( runTime.constant() ),
                                                           mesh,
                                                           ref.IOobject.MUST_READ_IF_MODIFIED,
                                                           ref.IOobject.NO_WRITE ) )

    force = U / ref.dimensionedScalar( ref.word( "dt" ), ref.dimTime, runTime.deltaTValue() )

    K = man.Kmesh( mesh )
    forceGen = man.UOprocess( K, runTime.deltaTValue(), turbulenceProperties )
    
    return turbulenceProperties, force, K, forceGen


#--------------------------------------------------------------------------------------
def globalProperties( runTime, U, nu, force ):
    ref.ext_Info() << "k(" << runTime.timeName()  << ") = " << 3.0 / 2.0 * U.magSqr().average().value() << ref.nl

    ref.ext_Info() << "epsilon(" << runTime.timeName() \
                   << ") = " << ( 0.5 * nu * ( ( ref.fvc.grad( U ) + ref.fvc.grad( U ).T() ).magSqr() ).average() ).value() << ref.nl

    ref.ext_Info() << "U.f(" << runTime.timeName() << ") = " << 181.0 * ( U & force ).average().value() << ref.nl
    pass


#--------------------------------------------------------------------------------------
def main_standalone( argc, argv ):

    args = ref.setRootCase( argc, argv )

    runTime = man.createTime( args )

    mesh = man.createMeshNoClear( runTime )
    
    transportProperties, nu = readTransportProperties( runTime, mesh )
    
    p, U, phi = _createFields( runTime, mesh )
    
    turbulenceProperties, force, K, forceGen = readTurbulenceProperties( runTime, mesh, U )
    
    cumulativeContErr = ref.initContinuityErrs()

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    ref.ext_Info() << "\nStarting time loop\n" 

    while runTime.loop():
        ref.ext_Info() << "Time = " << runTime.timeName() << ref.nl << ref.nl
       
        piso, nCorr, nNonOrthCorr, momentumPredictor, transonic, nOuterCorr = ref.readPISOControls( mesh )
       
        force.internalField() << ( ref.ReImSum( ref.fft.reverseTransform( K / ( K.mag() + 1.0e-6 ) ^ forceGen.newField(), K.nn() ) ) )
        
        globalProperties( runTime, U, nu, force )
        
        UEqn = ref.fvm.ddt( U ) + ref.fvm.div( phi, U ) - ref.fvm.laplacian( nu, U ) == force 

        ref.solve( UEqn == - man.fvc.grad( p ) )
        
        # --- PISO loop

        for corr  in range( 1 ):
            rUA = 1.0 / UEqn.A()

            U << rUA * UEqn.H()
            phi << ( ref.fvc.interpolate( U ) & mesh.Sf() ) + ref.fvc.ddtPhiCorr( rUA, U, phi )

            pEqn = ref.fvm.laplacian( rUA, p ) == ref.fvc.div( phi )

            pEqn.solve()

            phi -= pEqn.flux()

            cumulativeContErr = ref.ContinuityErrs( phi, runTime, mesh, cumulativeContErr )  

            U -= rUA * ref.fvc.grad( p )
            U.correctBoundaryConditions()
            pass

        runTime.write()
        
        if runTime.outputTime():
            ref.calcEk( U, K ).ext_write( ref.fileName( runTime.path() )/ref.fileName("graphs")/ref.fileName( runTime.timeName() ), 
                                          ref.word( "Ek" ), 
                                          runTime.graphFormat() )
            pass

        ref.ext_Info() << "ExecutionTime = " << runTime.elapsedCpuTime() << " s" \
                   << "  ClockTime = " << runTime.elapsedClockTime() << " s"  << ref.nl << ref.nl
        pass

    ref.ext_Info() << "End\n" << ref.nl 

    import os
    return os.EX_OK


#--------------------------------------------------------------------------------------
from Foam import FOAM_REF_VERSION
if FOAM_REF_VERSION( ">=", "020000" ):
   if __name__ == "__main__" :
      import sys, os
      argv = sys.argv
      os._exit( main_standalone( len( argv ), argv ) )
      pass
   pass
else:
   ref.ext_Info()<< "\nTo use this solver, It is necessary to SWIG OpenFoam2.0.0 or higher \n "     
   pass


#--------------------------------------------------------------------------------------

