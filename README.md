# CalcParsecIsoPars
Bayesian pipeline for calculating ages, distances, and other isochrone parameters.

User will need AstroMethods.py, DistFunc.py and CoordTrans.py from p.das@surrey.ac.uk. 

## ParsecIsochrones.py
Class that creates and pickles a dictionary of interpolants of Parsec isochrones. This has already been run, creating stellarprop_parsecdefault_currentmass_m0mugrizJHK.dill. If this needs to be rerun, the Parsec isochrones need to be downloaded from http://stev.oapd.inaf.it/cgi-bin/cmd.

## SpectroPhotoAstromSeismoDist.py
Class that needs installation of mwdust (https://github.com/jobovy/mwdust) and has location set within the code (this needs to be reset or set up separately). This is a class that reads in the isochrone interpolants, initializes the dust map, calculates a prior for the isochrone parameters, calculates the posterior probability on a grid of age, metallicity, mass, and distance, calculates the maximum posterior solution, and estimates the mean and standard deviation moments for age, metalllicty, mass, distance, logg, and teff.

## CreateMockIsochroneData.ipynb
Notebook that creates a mock dataset of l, b, parallax, metallicity, mass, teff, logg, and JHK apparent magnitudes and errors for testing the ability to derive age, metallicity, mass, and distance. 

## CalcDistIsoPars.ipynb
Notebook that uses SpectroPhotoAstromSeismoDist.py to calculate distances and isochrone parameters and uncertainties for a sample of stars.

