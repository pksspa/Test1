"""
Read VTK file and make a plot of the VTK file as a png
Create a LPF file from the VTK file
Run MODFLOW- steady state simulation
Save list file and heads file
Identify calibration target.
"""
import os
import sys 
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 


def AddToPyPath(pth):
    if pth not in sys.path:sys.path.append(pth)


#Add the MODFLOW Utils directory to the path
pth = 'C:/Projects/PhD/Inversion_of_CategoricalFields/MODFLOWUtils/'
AddToPyPath(pth)


from MODFLOWUtils import getLayRowCol

def readppfile(ppfileMaster):
    """
    Reads the contents of the pilot point file
    Returns a dictionary containing the pilotpoint data
    """
    ppdict={}
    """Read the contents of the pilot point file """
    with open(ppfileMaster,'r') as fin:
        for line in fin.readlines():
            ppname,x,y,zone,ppval = line.split()
            rec ={'x':np.double(x),'y':np.double(y),'zone':np.int(zone), 'ppval':np.double(ppval)}
            if ppname in ppdict:
                ppdict[ppname].update(rec)
            else:
                ppdict.update({ppname:rec})
        fin.close()     
    
    return ppdict


def GetRandomSeeds(randomseedfile):
    with open(randomseedfile,'r') as fin:
        fin.readline() # Skip the first line
        randomseeds=[int(line) for line in fin.readlines()] 
        fin.close()
    return randomseeds

def readvtkfile(vtkfile):
    """
    Reads a 2D Integer array from a VTK file and returns the array
    """
    with open(vtkfile, 'r')as fin:
        
        """Skip 4 lines """
        for i in range(1,5):
            fin.readline() 
             
        """Read the rows and cols """
        a,b,c,d =(fin.readline()).split()
        nx,ny = map(np.int,[b,c])
        
        """Read the origin """        
        a,b,c,d =(fin.readline()).split()
        x0,y0,z0=map(np.double,[b,c,d])
        
        """Read the cell discretization """
        a,b,c,d =(fin.readline()).split()
        delx,dely,delz = map(np.double,[b,c,d])
        
        """Skip 3 more lines """
        for i in range(1,4):
            fin.readline() 
        
        """Read the VTK array
        Flip the array vertically to fit MODFLOW convention"""
        zones=np.zeros((ny,nx),dtype='int') # MODFLOW : ROW/COL
        for i in range(ny-1,-1,-1):
            for j in range(0,nx,1):
                line =fin.readline()    
                zones[i,j]=np.double(line)
        fin.close()

    return nx,ny,delx,dely,zones

 
def writeformattedzones(zonearray,zoneoutfile):
    """
    Write a 2D Zone array to a formatted zones file.
    """
    assert(len(zonearray.shape)==2)
    nrow,ncol =zonearray.shape
    with open(zoneoutfile,'w') as fout:
        for i in range(0,nrow):
            line=''
            for j in range(0,ncol):
                line= line + '{:5}'.format(zonearray[i][j])
            fout.write(line +'\n') 
        
    fout.close()

def ModifyFieldgenSeed(fname,seed):
    """
    Replace the seed number in the fieldgen input file randomly
    """
    with open(fname,'r') as fin:
        linelist=[line for line in fin.readlines()]
        fin.close()
    
    #Change the line containing the random seed
    linelist[19]= str(seed) +'\n'
    
    with open(fname,'w') as fout:
        for i in range(len(linelist)):
            fout.write(linelist[i])
        fout.close()
def readarray(nrows,ncols,datatype,fname):
    """
    Reads a 2D array from an ASCII file and returns it
    """    
    if(datatype=='int'):
        array2d = np.zeros((nrows,ncols),dtype='int')
    elif(datatype=='double'):
        array2d = np.zeros((nrows,ncols),dtype='double')
    #listvals=[]
    irow=0
    icol=0
    with open(fname,'r') as fin:
        for line in fin.readlines():
            #print line
            if(datatype=='int'):
                listvals=[int(x) for x in line.split()]
            elif(datatype=='double'):
                listvals=[np.double(x) for x in line.split()]
                ''
            if(icol+len(listvals) > ncols):
                irow=irow+1
                icol=0
            for j in range(len(listvals)):
                #print irow,icol
                array2d[irow,icol+j]=listvals[j] 
            icol=icol+len(listvals)
                
        fin.close()
    
    
    return array2d

def plotarray(zonearray,Z,dirname,runid):
    
    """
    Plots HY properties
    """
    
    assert(len(Z.shape)==2)
    assert(len(zonearray.shape)==2)
    
    fig =plt.figure(figsize=(11,8.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.3)
    
    """Compute the log of HY """
    nrows,ncols=Z.shape
    logZ =np.log10(Z)
    logZ1d=np.reshape(logZ, (nrows*ncols,))
   
  
    """Histogram of log HY """  
    ax3=fig.add_subplot(2,1,1)
    mybins=np.arange(-3.0,4.0,0.333333)
    ax3.hist(logZ1d, bins=mybins, normed=0, facecolor='green', alpha=0.75)
    ax3.set_xlabel('Log10 HY')
    ax3.set_ylabel('Frequency')
    ax3.set_ylim(0,30000)
    ax3.grid(True)    
    
    """Plot of HY Zone Array """
    ax1=fig.add_subplot(2,2,3)
    im1 =ax1.imshow(zonearray,interpolation='bilinear')
    """
    create an axes on the right side of ax1. The width of cax will be 5%
    of ax and the padding between cax1 and ax1 will be fixed at 0.05 inch.
    """
    divider= make_axes_locatable(ax1)
    cax1=divider.append_axes("right",size="5%",pad=0.05)
    cbar1=plt.colorbar(im1, cax=cax1,ticks=[0,1])
    cbar1.ax.set_yticklabels(['Low','High']) #Vertically Oriented Colorbar
    ax1.set_title('HYZones'+"{0:04d}".format(runid))
       
    """Plot of HY Array"""
    ax =fig.add_subplot(2,2,4)
    im =ax.imshow(logZ,interpolation='bilinear',vmin=-2,vmax=2)
    """
    create an axes on the right side of ax. The width of cax will be 5%
    of ax and the padding between cax and ax will be fixed at 0.05 inch.
    """
    divider= make_axes_locatable(ax)
    cax=divider.append_axes("right",size="5%",pad=0.05)    
    #Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar=plt.colorbar(im, cax=cax,ticks=[-2,-1,0,1,2])
    cbar.ax.set_yticklabels(['0.01','0.1','1','10','>100']) #Vertically Oriented Colorbar
    ax.set_title('HYField'+"{0:04d}".format(runid))
    
    """Save the figure """
    fout =dirname +"/HY" + "{0:04d}".format(runid)+".png"
    plt.tight_layout()
    plt.savefig(fout)
    plt.clf()

def writeppfile(hyarray,ppdict,xspacing,yspacing,ppfileout):
    """
    Reads an array of hydraulic conductivities
    Reads a dictionary of pilot point data
    Reads the X and Y spacing arrays
    Writes an output pilot point file with the hydraulic conductivities at the pilot point locations
    """    
    zspacing=[0.0,-1.0]
    
    with open(ppfileout, 'w') as fout:    
        for pp in sorted(ppdict.keys()):
            ilay,irow,icol = getLayRowCol(ppdict[pp]['x'],ppdict[pp]['y'],0.0,xspacing,yspacing,zspacing)
            newppval = hyarray[irow,icol]
            fout.write(pp + '\t' + str(ppdict[pp]['x']) + '\t' + str(ppdict[pp]['y']) + '\t' + str(ppdict[pp]['zone']) + '\t' + str(newppval) + '\n' )
    fout.close()

def writearray(array,fname):
    """
    Write a 2D array to an output file in 10E12.4 format or 10I10 format
    """
    assert(len(array.shape)==2)
    nrow,ncol =array.shape
    lookupdict={}
    with open('C:/Projects/PhD/Inversion_of_CategoricalFields/Task1/hylookuptable.dat','r') as fin:
        for line in fin.readlines():
            key,val = line.split()
            #print key,val
            lookupdict.update({int(key):np.double(val)})  
    fin.close()
    #print lookupdict
    
    newarray = np.zeros((ny,nx),dtype='double')
    for key, val in lookupdict.iteritems(): newarray[array==key] = np.double(val)
    
    with open(fname,'w') as fout:
        for i in range(0,ny):
            jprev=0
            jnew=0
            while(jnew <ncol):
                if(jprev+12)> ncol :
                    jnew=ncol
                else:
                    jnew=jprev+12       
                line=''
                #print jnew,jprev
                for k in range(jprev,jnew):
                    line= line + '{:12.4e}'.format(newarray[i][k])
                
                jprev=jnew      
                fout.write(line +'\n') 
        
    fout.close()
    
    
def plothistogram(arrname,xtitle,outfile):
    """
    Make a histogram of arrname
    """
  
  
    
    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # the histogram of the data
    n, bins, patches = ax.hist(arrname, 10, normed=0, facecolor='green', alpha=0.75)
    
    # hist uses np.histogram under the hood to create 'n' and 'bins'.
    # np.histogram returns the bin edges, so there will be 10 probability
    # density values in n, 11 bin edges in bins and 10 patches.  
    
    ax.set_xlabel(xtitle)
    ax.set_ylabel('Frequency')
    ax.grid(True)    
    plt.savefig(outfile)
        

if __name__=='__main__':
    """ Main body of the program """
    nfiles=300
    fdir= 'C:/Projects/PhD/Inversion_of_CategoricalFields/Task0_CreateFields'
    fdir2= 'C:/Projects/PhD/Inversion_of_CategoricalFields/Task1'
    
    traveltimefile =open(fdir2+"/traveltimes.out",'w')
    traveltimes =np.zeros(nfiles,dtype='double')
    randomseeds=GetRandomSeeds(fdir2+"/RandomSeeds.dat")
    pathlistx=[]
    pathlisty=[]
    
    xspacing= np.arange(0.0,201.0,1.0)
    yspacing =np.arange(0.0,201.0,1.0)
    zspacing =np.array([0.0,-1.0])
    
    ppfileMaster = fdir2 + '/pilotpoints.dat'
    ppdict = readppfile(ppfileMaster)
    #print ppdict
    
    for i in range(0,nfiles):
        print i
        if(i<10):
            vtkfile=fdir + "/" + "test_sim"+"0000"+str(i)+".vtk"
        elif(i<100):
            vtkfile=fdir + "/" + "test_sim"+"000"+str(i)+".vtk"
        elif(i<1000):
            vtkfile= fdir + "/" + "test_sim"+"00"+str(i)+".vtk"
        """Read vtkfile"""
        #nx,ny,delx,dely,hy = readvtkfile(vtkfile)
        nx,ny,delx,dely,zones = readvtkfile(vtkfile)
        """Create Zones.inf"""
        writeformattedzones(zonearray=zones, zoneoutfile=fdir2+'/zones.inf')
        
        """Modify Fieldgen Seed"""        
        os.chdir(fdir2)
        ModifyFieldgenSeed(fname='fieldgen.in',seed=randomseeds[i])
        
        """"Use Fieldgen to create a random hy distribution"""
        os.system('fieldgensrc.exe < fieldgen.in')        
        
        """Read HY array"""
        hy=readarray(nrows=ny,ncols=nx,datatype='double',fname=fdir2 + '/hyarray1.ref')
        
        """Plot HY field"""
        #plotarray(hy,fout)
        fout =fdir2 +"/" + "HYField"+"000"+str(i)+".png"
        plotarray(zones,hy,dirname=fdir2,runid=i)
        
        """Write Pilot Point File """
        ppfileout = fdir2 +"/Pilotpoints_"+str(i) +".dat"
        writeppfile(hyarray=hy,ppdict=ppdict,xspacing=xspacing,yspacing=yspacing,ppfileout=ppfileout )
        #writearray(hy,fdir2+'/HYArray.ref') 
        
        
       
        """Run MODFLOW Model """
        os.system('mf2k_1_18_mst_dble.exe < mf2k.in')
        """Save files for future use """
        os.system('copy zones.inf zones_' + str(i) + '.inf')
        os.system('copy hyarray1.ref hyarray1_' + str(i) + '.ref')
        os.system('copy mod.hds mod_' + str(i) + '.hds')
        os.system('copy mod.lst mod_' + str(i) + '.lst')
        """Run MODPATH Model """
        os.system('Mpathr5_0.exe')
        newpathlinefile = 'pathline_' + str(i) + '.ptl'
        os.system('copy pathline ' + newpathlinefile )
        
        xp =[]
        yp=[]
        with open(newpathlinefile, 'r') as fin:
            fin.readline() #Skip one line
            for line in fin.readlines():
                a,b,c,d,e,f,g,h,i1,j1 = line.split()
                col=np.double(g)
                row =np.double(h)
                xp.append((col-1)*delx +0.5*delx)
                yp.append((ny-row)*dely+0.5*dely)
                
        traveltimefile.write(f+'\t'+ h+ '\t' + g +'\n')
        traveltimes[i] =np.double(f)
        
        pathlistx.append(xp)
        pathlisty.append(yp)
       
    traveltimefile.close()
    
    pathfig =plt.figure()
    ax =pathfig.add_subplot(111)
    plt.xlim(0,nx*delx)
    plt.ylim(0,ny*dely)
    for i in range(0,nfiles):
        l = ax.plot(pathlistx[i], pathlisty[i], '-', linewidth=1)
    plt.savefig('paths.png')
    plt.clf()
    
    plothistogram(traveltimes,'Traveltime(days)','traveltimeshistogram.png')    
    print'I am done'