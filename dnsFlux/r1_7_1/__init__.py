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
def readTransportProperties( runTime, mesh ):
    
    from Foam.OpenFOAM import IOdictionary, IOobject, word, fileName, ext_Info, nl
    ext_Info() << "Reading transportProperties\n" << nl
    
    transportProperties = IOdictionary( IOobject( word( "transportProperties" ),
                                                  fileName( runTime.constant() ),
                                                  mesh,
                                                  IOobject.MUST_READ,
                                                  IOobject.NO_WRITE ) )
    from Foam.OpenFOAM import dimensionedScalar
    
    nu = dimensionedScalar( transportProperties.lookup( word( "nu" ) ) )

    return transportProperties, nu


#----------------------------------------------------------------------------
def _createFields( runTime, mesh ):
    from Foam.OpenFOAM import ext_Info, nl
    from Foam.OpenFOAM import IOdictionary, IOobject, word, fileName
    from Foam.finiteVolume import volScalarField
        
    ext_Info() << "Reading field p\n" << nl
    p = volScalarField( IOobject( word( "p" ),
                                  fileName( runTime.timeName() ),
                                  mesh,
                                  IOobject.MUST_READ,
                                  IOobject.AUTO_WRITE ),
                        mesh )

    ext_Info() << "Reading field U\n" << nl
    from Foam.finiteVolume import volVectorField
    U = volVectorField( IOobject( word( "U" ),
                                  fileName( runTime.timeName() ),
                                  mesh,
                                  IOobject.MUST_READ,
                                  IOobject.AUTO_WRITE ),
                        mesh )

    from Foam.finiteVolume.cfdTools.incompressible import createPhi
    phi = createPhi( runTime, mesh, U )

    return p, U, phi
    

#--------------------------------------------------------------------------------------
def readTurbulenceProperties( runTime, mesh, U ):
    from Foam.OpenFOAM import IOdictionary, IOobject, word, fileName, ext_Info, nl
    
    ext_Info() << "Reading turbulenceProperties\n" << nl
    turbulenceProperties = IOdictionary( IOobject( word( "turbulenceProperties" ),
                                                   fileName( runTime.constant() ),
                                                   mesh,
                                                   IOobject.MUST_READ,
                                                   IOobject.NO_WRITE ) )

    from Foam.OpenFOAM import dimensionedScalar, dimTime
    force = U / dimensionedScalar( word( "dt" ), dimTime, runTime.deltaT().value() )

    from Foam.randomProcesses import Kmesh, UOprocess
    K = Kmesh( mesh )
    forceGen = UOprocess( K, runTime.deltaT().value(), turbulenceProperties )
    
    return turbulenceProperties, force, K, forceGen


#--------------------------------------------------------------------------------------
def globalProperties( runTime, U, nu, force ):
    from Foam.OpenFOAM import ext_Info, nl    
    from Foam import fvc
    ext_Info() << "k(" << runTime.timeName()  << ") = " << 3.0 / 2.0 * U.magSqr().average().value() << nl

    ext_Info() << "epsilon(" << runTime.timeName() << ") = " << ( 0.5 * nu * ( ( fvc.grad( U ) + fvc.grad( U ).T() ).magSqr() ).average() ).value() << nl

    ext_Info() << "U.f(" << runTime.timeName() << ") = " << 181.0 * ( U & force ).average().value() << nl
    pass


#--------------------------------------------------------------------------------------
def main_standalone( argc, argv ):

    from Foam.OpenFOAM.include import setRootCase
    args = setRootCase( argc, argv )

    from Foam.OpenFOAM.include import createTime
    runTime = createTime( args )

    from Foam.OpenFOAM.include import createMeshNoClear
    mesh = createMeshNoClear( runTime )
    
    transportProperties, nu = readTransportProperties( runTime, mesh )
    
    p, U, phi = _createFields( runTime, mesh )
    
    turbulenceProperties, force, K, forceGen = readTurbulenceProperties( runTime, mesh, U )
    
    from Foam.finiteVolume.cfdTools.general.include import initContinuityErrs
    cumulativeContErr = initContinuityErrs()

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
    from Foam.OpenFOAM import ext_Info, nl
    ext_Info() << "\nStarting time loop\n" 

    while runTime.loop():
        ext_Info() << "Time = " << runTime.timeName() << nl << nl
       
        from Foam.finiteVolume.cfdTools.general.include import readPISOControls
        piso, nCorr, nNonOrthCorr, momentumPredictor, transonic, nOuterCorr = readPISOControls( mesh )
       
        from Foam.randomProcesses import fft
        from Foam.OpenFOAM import ReImSum
        force.internalField().ext_assign( ReImSum( fft.reverseTransform( K / ( K.mag() + 1.0e-6 ) ^ forceGen.newField(), K.nn() ) ) )
        
        globalProperties( runTime, U, nu, force )
        
        from Foam import fvm
        UEqn = fvm.ddt( U ) + fvm.div( phi, U ) - fvm.laplacian( nu, U ) == force 
        
        from Foam import fvc
        from Foam.finiteVolume import solve
        solve( UEqn == -fvc.grad( p ) )
        
        # --- PISO loop

        for corr  in range( 1 ):
            rUA = 1.0 / UEqn.A()

            U.ext_assign( rUA*UEqn.H() )
            phi.ext_assign( ( fvc.interpolate( U ) & mesh.Sf() ) + fvc.ddtPhiCorr( rUA, U, phi ) )

            pEqn = fvm.laplacian( rUA, p ) == fvc.div( phi )

            pEqn.solve()

            phi.ext_assign( phi - pEqn.flux() )

            from Foam.finiteVolume.cfdTools.incompressible import continuityErrs
            cumulativeContErr = continuityErrs( mesh, phi, runTime, cumulativeContErr )

            U.ext_assign( U - rUA * fvc.grad( p ) )
            U.correctBoundaryConditions()
            pass

        runTime.write()
        
        if runTime.outputTime():
            from Foam.randomProcesses import calcEk
            from Foam.OpenFOAM import word, fileName
            calcEk( U, K ).write( fileName( runTime.timePath() / fileName( "Ek" ) ), runTime.graphFormat() )
            pass

        ext_Info() << "ExecutionTime = " << runTime.elapsedCpuTime() << " s" \
                   << "  ClockTime = " << runTime.elapsedClockTime() << " s"  << nl
        pass

    ext_Info() << "End\n" << nl 

    import os
    return os.EX_OK


#--------------------------------------------------------------------------------------
from Foam import FOAM_REF_VERSION
if FOAM_REF_VERSION( ">=", "010700" ):
   if __name__ == "__main__" :
      import sys, os
      argv = sys.argv
      os._exit( main_standalone( len( argv ), argv ) )
      pass
   pass
else:
   from Foam.OpenFOAM import ext_Info
   ext_Info()<< "\nTo use this solver, It is necessary to SWIG OpenFoam1.7.0 or higher \n "     
   pass


#--------------------------------------------------------------------------------------

