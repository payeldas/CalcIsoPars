"""
BAYESIAN METHOD TO CALCULATE MASSES, AGES, METALLICITIES AND DISTANCES FOR 
SPECTRO-PHOTO-ASTROM-SEISMO DATA
"""
import numpy as np
import CoordTrans as ct
import sys
import dill as dill
from scipy.integrate import dblquad,trapz,cumtrapz
from scipy.interpolate import interp1d
import AstroMethods as am
import DistFunc as df
# Create environment variable with dustmaps
import os
os.environ['DUST_DIR'] = "/home/payel/Dropbox/ResearchProjects/shared/dustmaps/"
import mwdust

class SpectroPhotoAstromSeismoDist:
    
    ## CLASS CONSTRUCTOR
    # solpos        - solar position                                         [array]
    # dust          - whether to calculate extinction using Jo Bovy's mwdust [double]
    # agemin        - minimum age of isochrone to consider (Gyr) [double]
    # agemax        - maximum age of isochrone to consider (Gyr) [double] 
    def  __init__(self,solpos,dust,agemin,agemax):
        self.solpos = np.copy(solpos)
        self.dust   = np.copy(dust)
        self.agemin = np.copy(agemin)
        self.agemax = np.copy(agemax)
        
        # Undill Parsec isochrones and interpolants
        print(" ")
        print("Undilling isochrones and interpolants...")
        with open("/home/payel/Dropbox/ResearchProjects/shared/python/ParsecIsochronesAgeDistance/data/stellarprop_parsecdefault_currentmass_m0mugrizJHK.dill", "rb") as input:
            self.pi = dill.load(input)
        print("...done.")
        print(" ")

        self.isoage = np.copy(self.pi.isoage) 
        self.isomh  = np.copy(self.pi.isomh)
        
        # Set limits for DF normalization
        self.universe_agemin = np.min(self.isoage)
        self.universe_agemax = np.max(self.isoage) # Age of the universe (Gyr)
        self.universe_mhmin  = np.min(self.isomh)
        self.universe_mhmax  = np.max(self.isomh)
        
        # Calculate galaxy model normalizations
        fbulge,fthick,fhalo = self.calcDfNorms()
        self.fbulge  = np.copy(fbulge)
        self.fthick  = np.copy(fthick)
        self.fhalo   = np.copy(fhalo)
        
        # Initialize Sale, 2014 dust map if desired and share with class
        if (dust==True):
            self.dustmapJ  = mwdust.Combined15(filter = '2MASS J')
            self.dustmapH  = mwdust.Combined15(filter = '2MASS H')
            self.dustmapKs = mwdust.Combined15(filter = '2MASS Ks')
            
    ## BULGE MODEL
    # tau - age                 [vector]
    # mh  - metallicity         [vector]
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized thin disc df
    def bulgedf(self,tau,mh,R,z):
        
        tau = np.atleast_1d(tau)
        mh  = np.atleast_1d(mh)
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
        
        # Metallicity df
        mumh  = -0.3       
        sigmh = 0.3
        pmh   = np.exp(-(mh-mumh)**2./(2.*sigmh**2.))/np.sqrt(2.*np.pi*sigmh**2.) 
        
        # Age df
        mutau  = 5.
        sigtau = 5.
        ptau  = np.exp(-(tau-mutau)**2./(2.*sigtau**2.))/np.sqrt(2.*np.pi*sigtau**2.) 
            
        # Position df
        q     = 0.5
        gamma = 0.0
        delta = 1.8
        r0    = 0.075
        rt    = 2.1
        m     = np.sqrt((R/r0)**2.+(z/(q*r0))**2.)
        pr    = ((1+m)**(gamma-delta))/(m**gamma) * np.exp(-(m*r0/rt)**2.)
        
        pbulge = pmh*ptau*pr
            
        return(pbulge)
           
    ## THIN DISC MODEL
    # tau - age                 [vector]
    # mh  - metallicity         [vector]
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized thin disc df
    def thindiscdf(self,tau,mh,R,z):
        
        tau = np.atleast_1d(tau)
        mh  = np.atleast_1d(mh)
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
        
        # Metallicity df
        mumh  = 0.0       
        sigmh = 0.2
        pmh   = np.exp(-(mh-mumh)**2./(2.*sigmh**2.))/np.sqrt(2.*np.pi*sigmh**2.) 
        
        # Age df
        tauF  = 8.
        taus  = 0.43
        index = tau <=10.
        ptau  = np.zeros_like(tau)
        ptau[index] = np.exp(tau[index]/tauF - taus/(self.agemax-tau[index]))
            
        # Position df
        Rd    = 2.6
        zd    = 0.3
        pr    = np.exp(-R/Rd-np.abs(z)/zd)
        
        pthin = pmh*ptau*pr
            
        return(pthin)
        
    ## THICK DISC MODEL
    # tau - age                 [vector]
    # mh  - metallicity         [vector]
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized thick disc df
    def thickdiscdf(self,tau,mh,R,z):  
        
        tau = np.atleast_1d(tau)
        mh  = np.atleast_1d(mh)
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
        
        # Metallicity df
        mumh  = -0.6       
        sigmh = 0.5
        pmh   = np.exp(-(mh-mumh)**2./(2.*sigmh**2.))/np.sqrt(2.*np.pi*sigmh**2.) 
        
        # Age df        
        tauF  = 8.
        taus  = 0.43
        index = tau>10.
        ptau  = np.zeros_like(tau)
        ptau[index] = np.exp(tau[index]/tauF - taus/(self.agemax-tau[index]))
            
        # Distance df
        Rd    = 3.6
        zd    = 0.9
        pr    = np.exp(-R/Rd-np.abs(z)/zd)
        
        pthick = pmh*ptau*pr
        
        return(pthick)
        
    ## STELLAR HALO MODEL
    # tau - age                 [vector]
    # mh  - metallicity         [vector]
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized stellar halo df    
    def stellarhalodf(self,tau,mh,R,z):
        
        tau = np.atleast_1d(tau)
        mh  = np.atleast_1d(mh)
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
        
        # Metallicity df
        mumh  = -1.6     
        sigmh = 0.5
        pmh   = np.exp(-(mh-mumh)**2./(2.*sigmh**2.))/np.sqrt(2.*np.pi*sigmh**2.) 
        
        # Age df        
        mutau  = 11.0
        sigtau = 2.0
        ptau  = np.exp(-(tau-mutau)**2./(2.*sigtau**2.))/np.sqrt(2.*np.pi*sigtau**2.) 
       
        # Distance df
        rs    = np.sqrt(R**2.+z**2.)
        pr    = rs**(-3.39)
        
        phalo = pmh*ptau*pr
            
        return(phalo)
        
    ## CALCULATE WEIGHTS ON THICK DISC AND STELLAR HALO FOR MILKY WAY
    # Returns fthick, fhalo    
    def calcDfNorms(self):
        
        solpos = np.copy(self.solpos)
        
        # Solar position
        R0 = solpos[0]
        z0 = solpos[1]
        
        # Df values at solar position marginalizd over ages and metallicities
        bulgedf = dblquad(self.bulgedf,
                          self.universe_mhmin,self.universe_mhmax,
                          lambda tau: self.universe_agemin, lambda tau: self.universe_agemax,
                          args=(R0,z0))[0]
        thindiscdf    = dblquad(self.thindiscdf,
                                self.universe_mhmin,self.universe_mhmax,
                                lambda tau: self.universe_agemin, lambda tau: self.universe_agemax,
                                args=(R0,z0))[0]            
        thickdiscdf   = dblquad(self.thickdiscdf,
                                self.universe_mhmin,self.universe_mhmax,
                                lambda tau: self.universe_agemin, lambda tau: self.universe_agemax,
                                args=(R0,z0))[0]            
        stellarhalodf = dblquad(self.stellarhalodf,
                                self.universe_mhmin,self.universe_mhmax,
                                lambda tau: self.universe_agemin, lambda tau: self.universe_agemax,
                                args=(R0,z0))[0]

        # Calcualte weights on galaxy components
        print("Calculating weights on galaxy components...")
        rbulgethin = 0.001        
        rthickthin = 0.15
        rhalothin  = 0.005
        fbulge     = rbulgethin*thindiscdf/bulgedf
        fthick     = rthickthin*thindiscdf/thickdiscdf
        fhalo      = rhalothin*thindiscdf/stellarhalodf
        print("     fbulge = "+str(np.round(fbulge,3)))
        print("     fthick = "+str(np.round(fthick,3)))
        print("     fhalo  = "+str(np.round(fhalo,3)))
        print("...done.")

        return(fbulge,fthick,fhalo)
        
    ## PRIOR PROBABILITY ON tau, mh, mass, s
    # modtau      - model ages [vector]
    # modmh       - model metallicities [vector]
    # modmass     - model masses [vector]
    # mods        - model distances [vector]
    # obsl        - Galactic longitude (rad) of observed star
    # obsb        - Galactic latitude (rad) of observed star
    # usegalprior - whether to use galaxy prior
    def pPrior(self,modtau,modmh,modmass,mods,obsl,obsb,usegalprior):
        
        # Copy shared variables
        solpos  = np.copy(self.solpos)
                
        # Define prior
        if (usegalprior):
            shape = np.shape(mods)
            R     = np.zeros_like(mods)
            z     = np.zeros_like(mods)
            if (len(shape)==1):
                xg = np.column_stack([obsl+mods*0.,obsb+mods*0.,mods])
                xp = ct.GalacticToPolar(xg,solpos)
                R  = xp[:,0]
                z  = xp[:,2]
            if (len(shape)==2):
                for j in range(shape[0]):
                    xg     = np.column_stack([obsl+mods[j,:]*0.,obsb+mods[j,:]*0.,mods[j,:]])
                    xp     = ct.GalacticToPolar(xg,solpos)
                    R[j,:] = xp[:,0]
                    z[j,:] = xp[:,2]
          
            # Jacobian p(x)dx = s**2p(s)ds
            jacdet = mods**2.
            prior = jacdet*\
                    df.kroupaImf(modmass)*(self.fbulge*self.bulgedf(modtau,modmh,R,z)+\
                    self.thindiscdf(modtau,modmh,R,z)+\
                    self.fthick*self.thickdiscdf(modtau,modmh,R,z)+\
                    self.fhalo*self.stellarhalodf(modtau,modmh,R,z))
        else:
            prior = np.ones_like(modmass)
               
        # Set prior to zero where parameters outside desired ranges (no need for mass)
        mhmin  = np.min(self.isomh)
        mhmax  = np.max(self.isomh)
        index = (modtau > self.agemax) & (modtau < self.agemin) & \
                (modmh > mhmax) & (modmh < mhmin)
        prior[index] = 0.
    
        return(prior)
            
    ## POSTERIOR PROBABILITY GRID FOR MODEL tau, mh, mass, s
    # obsStar     - dataframe with at least varpi, and possibly also actual mass, logTe, logg, J, H, Ks, and errors
    # obsl        - Galactic longitude (rad) of observed star
    # obsb        - Galactic latitude (rad) of observed star
    # nsig        - how many sigmas to extend metallicity grid over
    # nmass       - number of masses in grid
    # modsgrid    - model grid of distances
    # usegalprior - whether to use galaxy prior
    # Returns 4D grids for age, metallicity, mass, distance, and posterior probability, and maximum chi-squared solution
    def pPostGrid(self,obsStar,obsl,obsb,nsig,nmass,modsgrid,usegalprior):
        
        # Observations
        varpi,evarpi = obsStar["varpi"]
        mh,emh = obsStar["mh"]
        if ("mass" in obsStar.columns):
            mass,emass = obsStar["mass"]
        if ("teff" in obsStar.columns):
            teff,eteff = obsStar["teff"]
        if ("logg" in obsStar.columns):
            logg,elogg = obsStar["logg"]
        nmagcount = 0.
        if ("jmag" in obsStar.columns):
            nmagcount += 1
            appJ,eappJ = obsStar["jmag"]
        if ("hmag" in obsStar.columns):
            nmagcount += 1
            appH,eappH = obsStar["hmag"]
        if ("kmag" in obsStar.columns):
            nmagcount += 1
            appK,eappK = obsStar["kmag"]
            
        if ((nmagcount == 1) | (nmagcount == 2)):
            print("If apparent magnitudes are to be used, please provide all of J, H, and K!")
        else:
            jhk = True
              
        # Construct grids
        jagemin = np.sum(self.isoage<self.agemin)-1
        jagemax = np.sum(self.isoage<self.agemax)
        if (jagemin < 0):
            jagemin= 0
        if (jagemax == len(self.isoage)):
            jagemax = len(self.isoage)-1
        modagegrid  = self.isoage[jagemin:jagemax+1]
        nage        = len(modagegrid)
        
        mhmax      = mh+nsig*emh
        mhmin      = mh-nsig*emh
        jmhmin     = np.sum(self.isomh<mhmin)-1
        jmhmax     = np.sum(self.isomh<mhmax)
        if (jmhmin < 0):
            jmhmin= 0
        if (jmhmax == len(self.isomh)):
            jmhmax = len(self.isomh)-1
        modmhgrid  = self.isomh[jmhmin:jmhmax+1]
        nmh        = len(modmhgrid)
        ns         = len(modsgrid)
        modsmat    = np.vstack([modsgrid]*nmass)
        
        # Output matrices
        log10agepost = np.zeros([nage,nmh,nmass,ns])
        mhpost       = np.zeros([nage,nmh,nmass,ns])
        log10masspost= np.zeros([nage,nmh,nmass,ns])
        dmpost       = np.zeros([nage,nmh,nmass,ns])
        loggpost     = np.zeros([nage,nmh,nmass,ns])
        teffpost     = np.zeros([nage,nmh,nmass,ns])
        lnppost      = np.zeros([nage,nmh,nmass,ns])
  
        # Estimate extinction corrections if desired and ther are JHK apparent magnitudes
        if (jhk):
            if (self.dust==True):
                appJmat = appJ-self.dustmapJ(obsl/np.pi*180.,obsb/np.pi*180.,modsmat)
                appHmat = appH-self.dustmapH(obsl/np.pi*180.,obsb/np.pi*180.,modsmat)
                appKmat = appJ-self.dustmapKs(obsl/np.pi*180.,obsb/np.pi*180.,modsmat)
            else:
                appJmat = np.tile(appJ,[nmass,ns])
                appHmat = np.tile(appH,[nmass,ns])
                appKmat = np.tile(appK,[nmass,ns])
            # Calculate colour errors and matrix
            eJminK   = np.sqrt(eappJ**2. + eappK**2.)
            JminKmat = appJmat-appKmat  
                                     
        for jage in range(nage):
            for jmh in range(nmh):
                modtaumat  = np.zeros([nmass,ns]) + modagegrid[jage]
                modmhmat   = np.zeros([nmass,ns]) + modmhgrid[jmh]
                interpname = "age"+str(np.round(modagegrid[jage],8))+"mh"+str(modmhgrid[jmh])
                if (interpname in self.pi.isointerpdict):
                    isochrone = self.pi.isointerpdict[interpname]  
                else:
                    interpname = "age"+str(np.round(modagegrid[jage],9))+"mh"+str(modmhgrid[jmh])
                    if (interpname in self.pi.isointerpdict):
                        isochrone = self.pi.isointerpdict[interpname] 
                    else:
                        interpname = "age"+str(np.round(modagegrid[jage],10))+"mh"+str(modmhgrid[jmh])
                        if (interpname in self.pi.isointerpdict):
                            isochrone = self.pi.isointerpdict[interpname]  
                        else:
                            interpname = "age"+str(np.round(modagegrid[jage],11))+"mh"+str(modmhgrid[jmh])
                            if (interpname in self.pi.isointerpdict):
                                isochrone = self.pi.isointerpdict[interpname]  
                            else:
                                interpname = "age"+str(np.round(modagegrid[jage],12))+"mh"+str(modmhgrid[jmh])
                                if (interpname in self.pi.isointerpdict):
                                    isochrone = self.pi.isointerpdict[interpname]  
                                else:
                                    interpname   = "age"+str(np.round(modagegrid[jage],13))+"mh"+str(modmhgrid[jmh])
                                    if (interpname in self.pi.isointerpdict):
                                        isochrone = self.pi.isointerpdict[interpname]  
                                    else:
                                        interpname = "age"+str(np.round(modagegrid[jage],14))+"mh"+str(modmhgrid[jmh])                 
                                        
                isochrone   = self.pi.isointerpdict[interpname]
                massmin     = self.pi.massmindict[interpname]*1.01
                massmax     = self.pi.massmaxdict[interpname]*0.99
                massgrid    = np.logspace(np.log10(massmin),np.log10(massmax),nmass)
                modimassmat = np.column_stack([massgrid]*ns)
                
                # Likelihood of metallicity observable
                stellarpropmod = isochrone(modimassmat) # outputs initial mass, actual mass, log(Lum/Lsun), logTe,
                # log g, u, g, r, i, z, J, H, Ks
                modmassmat = stellarpropmod[1,:,:]    
                modteffmat = 10.**(stellarpropmod[3,:,:])
                modloggmat = stellarpropmod[4,:,:]
                modabsJmat = stellarpropmod[10,:,:]
                modabsHmat = stellarpropmod[11,:,:]
                modabsKmat = stellarpropmod[12,:,:]
                
                # Fill output arrays
                log10agepost[jage,jmh,:,:]  = np.log10(modtaumat)
                mhpost[jage,jmh,:,:]        = modmhmat
                log10masspost[jage,jmh,:,:] = np.log10(modmassmat)
                absJpost                    = am.absMag(appJmat,modsmat)
                dmpost[jage,jmh,:,:]        = appJmat-absJpost
                loggpost[jage,jmh,:,:]      = modloggmat
                teffpost[jage,jmh,:,:]      = modteffmat
                                                                                                    
                # Calculate prior
                pprior = self.pPrior(modtaumat,modmhmat,modmassmat,modsmat,obsl,obsb,usegalprior)
                
                # Calculate posterior probability
                lnprob = np.log(np.copy(pprior))
                
                # Parallax
                modvarpimat = 1./modsmat
                lnpparmat = -(modvarpimat-varpi)**2./(2.*evarpi**2.) - np.log(np.sqrt(2.*np.pi*evarpi**2.))
                lnprob += lnpparmat
                # Metallicity
                lnpmh = -(mh-modmhmat)**2./(2.*emh**2.) - np.log(np.sqrt(2.*np.pi*emh**2.))
                lnprob += lnpmh
                if ("mass" in obsStar.columns):
                    lnpmassmat = \
                        -(mass-modmassmat)**2./(2.*emass**2.) - np.log(np.sqrt(2.*np.pi*emass**2.))
                    lnprob += lnpmassmat   
                if ("teff" in obsStar.columns):
                    lnpteffmat = \
                        -(teff-modteffmat)**2./(2.*eteff**2.) - np.log(np.sqrt(2.*np.pi*eteff**2.))
                    lnprob += lnpteffmat                
                if ("logg" in obsStar.columns):
                    lnploggmat = \
                        -(logg-modloggmat)**2./(2.*elogg**2.) - np.log(np.sqrt(2.*np.pi*elogg**2.))
                    lnprob += lnploggmat
                if (jhk):
                    modappHmat = am.appMag(modabsHmat,modsmat)
                    lnpappmagmat = \
                        -(appHmat-modappHmat)**2./(2.*eappH**2.) - np.log(np.sqrt(2.*np.pi*eappH**2.))
                    modJminKmat = modabsJmat-modabsKmat
                    lnpcolmat = \
                        -(JminKmat-modJminKmat)**2./(2.*eJminK**2.) - np.log(np.sqrt(2.*np.pi*eJminK**2.))
                    lnprob += lnpappmagmat+lnpcolmat
    
                lnppost[jage,jmh,:,:] = np.copy(lnprob)
                                           
        return(log10agepost,mhpost,log10masspost,dmpost,loggpost,teffpost,lnppost)
    
    ## ESTIMATE MAXIMUM POSTERIOR SOLUTION
    # log10agepost - 4D grid (nage*nmh*nmass*ns)
    # mhpost       - 4D grid (nage*nmh*nmass*ns)
    # log10masspost- 4D grid (nage*nmh*nmass*ns)
    # dmpost       - 4D grid (nage*nmh*nmass*ns)
    # loggpost     - 4D grid (nage*nmh*nmass*ns)
    # teffpost     - 4D grid (nage*nmh*nmass*ns)
    # lnppost      - 4D probability grid (nage*nmh*nmass*ns)
    # Returns parameters corresponding to maximum posterior solution
    def estMaxPost(self,log10agepost,mhpost,log10masspost,dmpost,loggpost,teffpost,lnppost):
        
        index          = (lnppost == np.max(lnppost))
        log10age_best  = log10agepost[index][0]
        mh_best        = mhpost[index][0]
        log10mass_best = log10masspost[index][0]
        dm_best        = dmpost[index][0]
        logg_best      = loggpost[index][0]
        teff_best      = teffpost[index][0]
        
        best_post_soln_dict = {"log10age_best": log10age_best,
                               "mh_best": mh_best,
                               "log10mass_best": log10mass_best,
                               "dm_best": dm_best,
                               "logg_best": logg_best,
                               "teff_best": teff_best}
        
        return(best_post_soln_dict)
    
    ## CALCULATE MOMENTS
    # log10agepost,mhpost,log10masspost,dmpost,loggpost,teffpost - 4D grid (nage*nmh*nmass*ns)
    # lnppost            - 4D posterior grid (nage*nmh*nmass*ns)
    # which_moments_dict - dictionary with which moments to calculate
    # Returns moments for arrays parsed
    def calcMoments(self,log10agepost,mhpost,log10masspost,dmpost,loggpost,teffpost,lnppost,which_moments_dict):
        
        nage,nmh,nmass,ns = np.shape(log10agepost)
        log10age          = log10agepost[:,0,0,0]
        mh                = mhpost[0,:,0,0]
        log10mass         = log10masspost[0,0,:,0]
        dm                = dmpost[0,0,0,:]
        
        # Ensure lowest non-zero lnppost doesn't exponentiate to zero
        #index          = (lnppost < sys.float_info.min)
        #if (np.sum(index)) > 0:
        #    addvalue = sys.float_info.min - np.min(lnppost)
        #    lnppost += addvalue
            
        # Define posterior
        ppost = np.exp(lnppost)
        
        # Define normalization arrays
        norm_age_mh_mass        = np.zeros([nage,nmh,nmass])
        norm_age_mh             = np.zeros([nage,nmh])
        norm_age                = np.zeros([nage])
        
        # Define age arrays
        log10age_mu_age_mh_mass = np.zeros([nage,nmh,nmass])
        log10age_2_age_mh_mass  = np.zeros([nage,nmh,nmass])
        log10age_mu_age_mh      = np.zeros([nage,nmh])
        log10age_2_age_mh       = np.zeros([nage,nmh])
        log10age_mu_age         = np.zeros([nage])
        log10age_2_age          = np.zeros([nage])
        
        # Define metallicity arrays if desired
        if which_moments_dict['mh']:
            mh_mu_age_mh_mass = np.zeros([nage,nmh,nmass])
            mh_2_age_mh_mass  = np.zeros([nage,nmh,nmass])
            mh_mu_age_mh      = np.zeros([nage,nmh])
            mh_2_age_mh       = np.zeros([nage,nmh])
            mh_mu_age         = np.zeros([nage])
            mh_2_age          = np.zeros([nage])
        
        # Define mass arrays if desired
        if which_moments_dict['mass']:
            log10mass_mu_age_mh_mass = np.zeros([nage,nmh,nmass])
            log10mass_2_age_mh_mass  = np.zeros([nage,nmh,nmass])
            log10mass_mu_age_mh      = np.zeros([nage,nmh])
            log10mass_2_age_mh       = np.zeros([nage,nmh])
            log10mass_mu_age         = np.zeros([nage])
            log10mass_2_age          = np.zeros([nage])
        
        # Define distance modulus arrays if desired
        if which_moments_dict['dm']:
            dm_mu_age_mh_mass = np.zeros([nage,nmh,nmass])
            dm_2_age_mh_mass  = np.zeros([nage,nmh,nmass])
            dm_mu_age_mh      = np.zeros([nage,nmh])
            dm_2_age_mh       = np.zeros([nage,nmh])
            dm_mu_age         = np.zeros([nage])
            dm_2_age          = np.zeros([nage])
        
        # Define logg arrays
        if which_moments_dict['logg']:
            logg_mu_age_mh_mass = np.zeros([nage,nmh,nmass])
            logg_2_age_mh_mass  = np.zeros([nage,nmh,nmass])
            logg_mu_age_mh      = np.zeros([nage,nmh])
            logg_2_age_mh       = np.zeros([nage,nmh])
            logg_mu_age         = np.zeros([nage])
            logg_2_age          = np.zeros([nage])
        
        # Define teff arrays
        if which_moments_dict['teff']:
            teff_mu_age_mh_mass = np.zeros([nage,nmh,nmass])
            teff_2_age_mh_mass  = np.zeros([nage,nmh,nmass])
            teff_mu_age_mh      = np.zeros([nage,nmh])
            teff_2_age_mh       = np.zeros([nage,nmh])
            teff_mu_age         = np.zeros([nage])
            teff_2_age          = np.zeros([nage])
        
        # Marginalize posterior probability over DM, mass, metallicity, and age to calculate various moments
        for jage in range(nage):
            for jmh in range(nmh):
                for jmass in range(nmass):
                    norm_age_mh_mass[jage,jmh,jmass]         = trapz(ppost[jage,jmh,jmass,:],dm)
                    log10age_mu_age_mh_mass[jage,jmh,jmass]  = trapz(log10agepost[jage,jmh,jmass,:]*ppost[jage,jmh,jmass,:],dm)
                    log10age_2_age_mh_mass[jage,jmh,jmass]   = trapz(log10agepost[jage,jmh,jmass,:]**2.*ppost[jage,jmh,jmass,:],dm)
                    
                    if which_moments_dict['mh']:
                        mh_mu_age_mh_mass[jage,jmh,jmass]        = trapz(mhpost[jage,jmh,jmass,:]*ppost[jage,jmh,jmass,:],dm)
                        mh_2_age_mh_mass[jage,jmh,jmass]         = trapz(mhpost[jage,jmh,jmass,:]**2.*ppost[jage,jmh,jmass,:],dm)
                    
                    if which_moments_dict['mass']:
                        log10mass_mu_age_mh_mass[jage,jmh,jmass] = trapz(log10masspost[jage,jmh,jmass,:]*ppost[jage,jmh,jmass,:],dm)
                        log10mass_2_age_mh_mass[jage,jmh,jmass]  = trapz(log10masspost[jage,jmh,jmass,:]**2.*ppost[jage,jmh,jmass,:],dm)
                    
                    if which_moments_dict['dm']:
                        dm_mu_age_mh_mass[jage,jmh,jmass]        = trapz(dmpost[jage,jmh,jmass,:]*ppost[jage,jmh,jmass,:],dm)
                        dm_2_age_mh_mass[jage,jmh,jmass]         = trapz(dmpost[jage,jmh,jmass,:]**2.*ppost[jage,jmh,jmass,:],dm)
                    
                    if which_moments_dict['logg']:
                        logg_mu_age_mh_mass[jage,jmh,jmass]      = trapz(loggpost[jage,jmh,jmass,:]*ppost[jage,jmh,jmass,:],dm)
                        logg_2_age_mh_mass[jage,jmh,jmass]       = trapz(loggpost[jage,jmh,jmass,:]**2.*ppost[jage,jmh,jmass,:],dm)
                    
                    if which_moments_dict['teff']:
                        teff_mu_age_mh_mass[jage,jmh,jmass]      = trapz(teffpost[jage,jmh,jmass,:]*ppost[jage,jmh,jmass,:],dm)
                        teff_2_age_mh_mass[jage,jmh,jmass]       = trapz(teffpost[jage,jmh,jmass,:]**2.*ppost[jage,jmh,jmass,:],dm)
                    
                norm_age_mh[jage,jmh]         = trapz(norm_age_mh_mass[jage,jmh,:],log10mass)
                log10age_mu_age_mh[jage,jmh]  = trapz(log10age_mu_age_mh_mass[jage,jmh,:],log10mass)
                log10age_2_age_mh[jage,jmh]   = trapz(log10age_2_age_mh_mass[jage,jmh,:],log10mass)
                
                if which_moments_dict['mh']:
                    mh_mu_age_mh[jage,jmh]        = trapz(mh_mu_age_mh_mass[jage,jmh,:],log10mass)
                    mh_2_age_mh[jage,jmh]         = trapz(mh_2_age_mh_mass[jage,jmh,:],log10mass)
                
                if which_moments_dict['mass']:
                    log10mass_mu_age_mh[jage,jmh] = trapz(log10mass_mu_age_mh_mass[jage,jmh,:],log10mass)
                    log10mass_2_age_mh[jage,jmh]  = trapz(log10mass_2_age_mh_mass[jage,jmh,:],log10mass)
                
                if which_moments_dict['dm']:
                    dm_mu_age_mh[jage,jmh]        = trapz(dm_mu_age_mh_mass[jage,jmh,:],log10mass)
                    dm_2_age_mh[jage,jmh]         = trapz(dm_2_age_mh_mass[jage,jmh,:],log10mass)
                
                if which_moments_dict['logg']:
                    logg_mu_age_mh[jage,jmh]      = trapz(logg_mu_age_mh_mass[jage,jmh,:],log10mass)
                    logg_2_age_mh_mass[jage,jmh]  = trapz(logg_2_age_mh_mass[jage,jmh,:],log10mass)
                
                if which_moments_dict['teff']:
                    teff_mu_age_mh[jage,jmh]      = trapz(teff_mu_age_mh_mass[jage,jmh,:],log10mass)
                    teff_2_age_mh[jage,jmh]       = trapz(teff_2_age_mh_mass[jage,jmh,:],log10mass)
                
            norm_age[jage]         = trapz(norm_age_mh[jage,:],mh)
            log10age_mu_age[jage]  = trapz(log10age_mu_age_mh[jage,:],mh)
            log10age_2_age[jage]   = trapz(log10age_2_age_mh[jage,:],mh)
            
            if which_moments_dict['mh']:
                mh_mu_age[jage]        = trapz(mh_mu_age_mh[jage,:],mh)
                mh_2_age[jage]         = trapz(mh_2_age_mh[jage,:],mh)
            
            if which_moments_dict['mass']:
                log10mass_mu_age[jage] = trapz(log10mass_mu_age_mh[jage,:],mh)
                log10mass_2_age[jage]  = trapz(log10mass_2_age_mh[jage,:],mh)
            
            if which_moments_dict['dm']:
                dm_mu_age[jage]        = trapz(dm_mu_age_mh[jage,:],mh)
                dm_2_age[jage]         = trapz(dm_2_age_mh[jage,:],mh)
            
            if which_moments_dict['logg']:
                logg_mu_age[jage]      = trapz(logg_mu_age_mh[jage,:],mh)
                logg_2_age[jage]       = trapz(logg_2_age_mh[jage,:],mh)
            
            if which_moments_dict['teff']:
                teff_mu_age[jage]      = trapz(teff_mu_age_mh[jage,:],mh)
                teff_2_age[jage]       = trapz(teff_2_age_mh[jage,:],mh)
            
        norm          = trapz(norm_age,log10age)
        
        log10age_mu   = trapz(log10age_mu_age,log10age)/norm
        log10age_2    = trapz(log10age_2_age,log10age)/norm
        log10age_std  = np.sqrt(log10age_2-log10age_mu**2.)
        moments_dict = {"log10age_mu":log10age_mu,
                        "log10age_std":log10age_std}
        
        if which_moments_dict['mh']:
            mh_mu         = trapz(mh_mu_age,log10age)/norm
            mh_2          = trapz(mh_2_age,log10age)/norm
            mh_std        = np.sqrt(mh_2-mh_mu**2.)
            moments_dict["mh_mu"]  = mh_mu
            moments_dict["mh_std"] = mh_std
                        
        if which_moments_dict['mass']:
            log10mass_mu  = trapz(log10mass_mu_age,log10age)/norm
            log10mass_2   = trapz(log10mass_2_age,log10age)/norm
            log10mass_std = np.sqrt(log10mass_2-log10mass_mu**2.)
            moments_dict["log10mass_mu"]  = log10mass_mu
            moments_dict["log10mass_std"] = log10mass_std
               
        if which_moments_dict['dm']:
            dm_mu         = trapz(dm_mu_age,log10age)/norm
            dm_2          = trapz(dm_2_age,log10age)/norm
            dm_std        = np.sqrt(dm_2-dm_mu**2.)
            moments_dict["dm_mu"]  = dm_mu
            moments_dict["dm_std"] = dm_std
               
        if which_moments_dict['logg']:
            logg_mu       = trapz(logg_mu_age,log10age)/norm
            logg_2        = trapz(logg_2_age,log10age)/norm
            logg_std      = np.sqrt(logg_2-logg_mu**2.)
            moments_dict["logg_mu"]  = logg_mu
            moments_dict["logg_std"] = logg_std
           
        if which_moments_dict['teff']:
            teff_mu       = trapz(teff_mu_age,log10age)/norm
            teff_2        = trapz(teff_2_age,log10age)/norm
            teff_std      = np.sqrt(teff_2-teff_mu**2.)
            moments_dict["teff_mu"]  = teff_mu
            moments_dict["teff_std"] = teff_std
      
        if (np.isnan(log10age_mu)):
            print("NaN age! Likely observed errors e.g. on logg are too small. Either increase them or use maximum posterior estimate.")
        
        return(moments_dict) 
        