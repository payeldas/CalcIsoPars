"""
CREATES AND PICKLES A DICTIONARY OF INTERPOLANTS OF PARSEC ISOCHRONES
Parsec isochrones give apparent magnitudes at fiducial distance of 10 pc and 
assume a Chabrier IMF
"""

import numpy as np
import os.path
from scipy.interpolate import interp1d
import dill as dill

class ParsecIsochrones:   
     
    ## CLASS INITIALIZER
    # isodir      - directory with isochrones
    # initialmass - whether to use initialmass or current mass to create interpolant
    def  __init__(self,isodir,initialmass):
        
        # Whether initial mass
        if (initialmass==True):
            imass=2
        else:
            imass=3
            
        # Number of metallicities
        nmh = 57
        
        # Number of ages, minimum age, maximum age
        nage   = 353
        log10agemin = 6.60
        log10agemax = 10.12
        
        # Metallicity prefix in filenames
        isomhprefix = ["Z0.00010","Z0.00011","Z0.00013","Z0.00014","Z0.00016",
                       "Z0.00018","Z0.00020","Z0.00022","Z0.00025","Z0.00028",
                       "Z0.00032","Z0.00036","Z0.00040","Z0.00048","Z0.00050",
                       "Z0.00056","Z0.00064","Z0.00071","Z0.00080","Z0.00089",
                       "Z0.00100","Z0.00112","Z0.00130","Z0.00141","Z0.00160",
                       "Z0.00178","Z0.00200","Z0.00234","Z0.00250","Z0.00282",
                       "Z0.00320","Z0.00355","Z0.00400","Z0.00447","Z0.00500",
                       "Z0.00562","Z0.00640","Z0.00708","Z0.00800","Z0.00891",
                       "Z0.01000","Z0.01220","Z0.01300","Z0.01410","Z0.01600",
                       "Z0.01780","Z0.02000","Z0.02240","Z0.02500","Z0.02820",
                       "Z0.03200","Z0.03550","Z0.04000","Z0.04470","Z0.05000",
                       "Z0.05620","Z0.06000"]
              
        # Isochrone metallicities [M/H]
        isomh = np.array([-2.192,-2.142,-2.078,-2.042,-1.987,-1.941,-1.890,
                          -1.841,-1.794,-1.743,-1.686,-1.641,-1.589,-1.541,
                          -1.492,-1.441,-1.385,-1.341,-1.288,-1.241,-1.190,
                          -1.141,-1.076,-1.040,-0.985,-0.939,-0.888,-0.819,
                          -0.790,-0.737,-0.681,-0.636,-0.583,-0.534,-0.485,
                          -0.433,-0.375,-0.330,-0.276,-0.227,-0.175,-0.124,
                          -0.056,-0.020,0.039,0.088,0.142,0.197,0.249,0.307,
                          0.369,0.421,0.481,0.539,0.597,0.660,0.696])
                           
        # Convert isochrone ages to Gyr
        isoage = (10.**np.linspace(log10agemin,log10agemax,nage))/1.e09
        
        print("Creating dictonary of interpolants of all relevant isochrones")
                
        # Initialize dictionaries
        isodict      = {}          
        isointerpdict= {}
        massmindict  = {}
        massmaxdict  = {}         
        for jage in range(nage):
            for jmh in range(nmh):
                isochrone = self.readIso(isoage[jage],isomh[jmh],isomhprefix[jmh],isodir)
                massmin   = np.min(isochrone[:,imass])
                massmax   = np.max(isochrone[:,imass])
                print("Minimum mass on relevant part of isochrone is " + str(massmin) + " and maximum mass is " + str(massmax))
                isointerp = self.createInterp(isochrone,isodir,initialmass)
               
                   # Name interpolant
                interpname                = "age"+str(np.round(isoage[jage],8))+\
                                            "mh"+str(np.round(isomh[jmh],3))
                isodict[interpname]       = isochrone
                isointerpdict[interpname] = isointerp
                massmindict[interpname]   = np.copy(massmin)
                massmaxdict[interpname]   = np.copy(massmax)
                                
        # Share variables
        self.isoage        = np.copy(isoage)
        self.isomh         = np.copy(isomh)
        self.isomhprefix   = np.copy(isomhprefix)
        self.isodict       = isodict
        self.isointerpdict = isointerpdict
        self.massmindict   = massmindict
        self.massmaxdict   = massmaxdict
                
        return
        
    ## CALCULATE INTRINSIC PROPERTIES OF STAR AT ANY AGE, METALLICITY, AND MASS
    ## USING INTERPOLANTS (CAN ALSO ACCESS ISOCHRONES AND INTTERPOLANTS DIRECTLY)
    # age  - age in Gyr
    # mh   - metallicity in dex
    # mass - mass in solar masses [double]
    # Returns initial mass, current mass, log L/L0, logTe, logg, u, g, r, i, z, J, H, Ks
    def __call__(self,age,mh,mass):  
        
        isoage = np.copy(self.isoage)
        isomh  = np.copy(self.isomh)
        
        # Find isochrone with closest age and metallicity
        agediff    = np.abs(isoage-age)
        ageiso     = isoage[agediff==np.min(agediff)][0]
        mhdiff     = np.abs(isomh-mh)
        mhiso      = isomh[mhdiff==np.min(mhdiff)][0]
        interpname = "age"+str(np.round(ageiso,8))+\
                     "mh"+str(np.round(mhiso,3))
        isochrone  = self.isointerpdict[interpname]
        massmin    = self.massmindict[interpname]
        massmax    = self.massmaxdict[interpname]
        
        if (np.any(mass<massmin)|np.any(mass>massmax)):
            print("Some masses are beyond the range of the isochrone! Reconsider...")
            print(" ")
            return(np.inf)

        return(isochrone(mass))
        
    ## READ ISOCHRONE
    # age      - age of isochrone
    # mh       - metallicity
    # mhprefix - prefix for filename
    # isofile  - isochrone filename
    # Returns isochrone
    def readIso(self,age,mh,mhprefix,isodir):
        
         # Isochrone filename                
        isofile = isodir+"/2mass_sdss"+mhprefix
        # Read appropriate isochrone
        print("Reading isochrone with [M/H] = " + str(mh) + " dex and age = "+ str(np.round(age,4)) + " Gyrs, in file "+isofile)
        
        isochrone=[]
        if (os.path.isfile(isofile)):    
            f = open(isofile, 'r')
            for line in f.readlines():
                if not line[0] in ['#']:
                    linearray = line.split()
                    isochrone.append(linearray)
        else:
            print("File does not exist!")
            print(isofile)
            print(" ")
        isochrone      = np.array(isochrone,dtype='S32')
        nentries,ncols = np.shape(isochrone)
        isochrone      = isochrone.astype(np.float)
        print("Read "+str(nentries)+ " entries, keeping "+str(ncols)+" columns of data.")
  
        # Select part of isochrone with correct age, create and share interpolant
        isochrone[:,1] = np.round(isochrone[:,1],2)
        index          = isochrone[:,1]==np.round(np.log10(age*1.e09),2)
        isochrone      = isochrone[index,:]
         
        return(isochrone)
                 
    ## CREATE INTERPOLANT OF ISOCHRONE 
    # isochrone   - isochrone array
    # isodir      - name of directory with isochrones
    # initialmass - True,False, whether to use initial or actual mass (as a result of mass loss) [logical]
    # Returns interpolant
    def createInterp(self,isochrone,isodir,initialmass):
        
        # Whether initial mass
        if (initialmass==True):
            imass=2
        else:
            imass=3

        # Select part of isochrone with correct age, create and share interpolant
        stellarprop    = np.column_stack([isochrone[:,2],       # Initial mass
                                          isochrone[:,3],       # Current mass
                                          isochrone[:,4],       # log L/L0
                                          isochrone[:,5],       # log Teff
                                          isochrone[:,6],       # log g
                                          isochrone[:,8],       # u
                                          isochrone[:,9],       # g
                                          isochrone[:,10],      # r
                                          isochrone[:,11],      # i
                                          isochrone[:,12],      # z
                                          isochrone[:,13],      # J
                                          isochrone[:,14],      # H
                                          isochrone[:,15]])     # Ks
  
        # Create interpolant    
        print("Creating interpolant of isochrone...")
        isointerp  = interp1d(isochrone[:,imass],np.transpose(stellarprop),"linear")
        
        print("Done.")
        print(" ")
        
        return(isointerp)

#%% TESTING
# Create isochrone class
isodir      = "../data/isochrones/parsec_isochrones_default"  
initialmass = False
pi = ParsecIsochrones(isodir,initialmass)

#%% Predict magnitudes and colours at some age, metallicity, and mass
age  = 0.004
mh   = -2.15
mass = 0.8
out  = pi(age,mh,mass)
print(out)

#%% Dill dictionary of isochrone interpolants
print("Dill dictionary of isochrone interpolants...")
with open("stellarprop_parsecdefault_currentmass_m0mugrizJHK.dill", "wb") as output:
    dill.dump(pi, output, dill.HIGHEST_PROTOCOL)
print("...done.")
print(" ")

#%% Test dill
print("Undilling isochrone interpolants...")
with open("stellarprop_parsecdefault_currentmass_m0mugrizJHK.dill", "rb") as input:
    pi_dill = dill.load(input)
print("...done.")

#%% Predict magnitudes and colours at some age, metallicity, and mass with dilled version
age  = 0.004
mh   = -2.15
mass = 0.8
out  = pi_dill(age,mh,mass)
print(out)
