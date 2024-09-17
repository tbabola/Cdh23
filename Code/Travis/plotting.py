import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.patches as patches
from skimage import io, filters

def getColors():
    color_dict = {'Thy1-GC6s; Cdh23 (Ahl/ahl)':"#DD0000",
                  'Thy1-GC6s; Cdh23 (ahl/ahl)':'#000000',
                  '(F1) Thy1-GC6s; Cdh23 (Ahl/ahl)':'#FFCC00'}       
    return color_dict

def getOrder():
    order = ['Thy1-GC6s; Cdh23 (ahl/ahl)',
             'Thy1-GC6s; Cdh23 (Ahl/ahl)',
             '(F1) Thy1-GC6s; Cdh23 (Ahl/ahl)']
    return order

def getMap(mean_dfof, percentile = 99, gain = [1.5, 1, 2], frameToTake = 45, freqToTake = [4,2,0], attenLevel = 1):
    map_image = np.zeros(shape=(mean_dfof.shape[0],mean_dfof.shape[1], mean_dfof.shape[2],3))
    for i, freq in enumerate(freqToTake):
        map_image[:,:,:,i] = mean_dfof[:,:,:,freq,atten_level]

    image = filters.gaussian(np.nanmean(map_image[frameToTake-2:frameToTake+2,:,:,:],axis=0),0.5)
    #normalize 
    percentile = np.percentile(image,q=percentile,axis=(0,1))
    percentile = np.max(percentile)
    image[:,:,0] = image[:,:,0] * gain[0] / (percentile*1.2)
    image[:,:,1] = image[:,:,1] * gain[1]/ (percentile*1.2)
    image[:,:,2] = image[:,:,2] * gain[2] / (percentile*1.2)
    image[image > 1] = 1
    image[image < 0] = 0

    return image


###function that plots FRA traces with individual trials or animals in grey, and average in black
def plotFRAtraces(array, ymin = -0.01, ymax=0.01, xmin = 0, xmax = 100, ylabels = [90,70,50,30], xlabels = [4,8,16,32,64], 
    convolve_len = 1, mean_color= [0,0,0], mean_to_ind_alpha = None, toneWindow = None):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['svg.fonttype'] = 'none'
    numcols = array.shape[1]
    numrows = array.shape[2]
    numanimals = array.shape[3]
    fig, axs = plt.subplots(ncols=numcols, nrows=numrows)
    print(numanimals)
    for k in range(numanimals):
        count = 1
        for j in range(numrows):
            for i in range(numcols):
                if not mean_to_ind_alpha:
                    axs[j,i].plot(np.convolve(array[xmin:xmax,i,j,k], np.ones(shape=(convolve_len,))/convolve_len, mode='same'),c=[0.9,0.9,0.9], linewidth = 0.5)
                else:
                    axs[j,i].plot(np.convolve(array[xmin:xmax,i,j,k], np.ones(shape=(convolve_len,))/convolve_len, mode='same'),c=mean_color, alpha= mean_to_ind_alpha, linewidth = 0.5)
                axs[j,i].patch.set_alpha(0.0)
                axs[j,i].set_xlim((xmin,xmax))
                axs[j,i].set_ylim((ymin,ymax))
                axs[j,i].axis("off")
                axs[j,i].set_xticks([])
                axs[j,i].set_yticks([])

                if toneWindow:
                    # Create a Rectangle patch
                    rect = patches.Rectangle((toneWindow[0], -100), toneWindow[1]-toneWindow[0], 200)#, linewidth=0, facecolor='gray', alpha=0.2,zorder=1)
                    # Add the patch to the Axes
                    axs[j,i].add_patch(rect)

                if i == 0 or j == (numrows-1):
                    axs[j,i].axis("on")
                    axs[j,i].spines['bottom'].set_visible(False)
                    axs[j,i].spines['top'].set_visible(False)
                    axs[j,i].spines['right'].set_visible(False)
                    axs[j,i].spines['left'].set_visible(False)
                    
                    if i == 0:
                        axs[j,i].set_yticks(np.array([0]), labels=[str(ylabels[j])], fontsize=6)
                        axs[j,i].tick_params(axis='y', which='major', pad=1, length = 2)
                    if j == (numrows-1):
                        axs[j,i].set_xticks(np.array([(xmax+xmin)/2]), labels=[str(xlabels[i])], fontsize=6)
                        axs[j,i].tick_params(axis='x', which='major', pad=1, length = 2)
                   

                # if (i == numcols and j == 0):
                #      axs[j,i].axis("on")

    ##this is a little convoluted but had to be done because subplots have their own layering, this code makes new
    ##subplots on top so individual traces don't overlap the mean
    count = 1
    for j in range(numrows):
        for i in range(numcols):
            ax = fig.add_subplot(numrows,numcols,count)
            ax.plot(np.convolve(np.nanmean(array[xmin:xmax,i,j,:],axis=1),np.ones(convolve_len,)/convolve_len,mode='same'),c=mean_color, linewidth=1)
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.patch.set_alpha(0.0)
            ax.set_xlim((xmin,xmax))
            ax.set_ylim((ymin,ymax))
            #print("shpae", array[:,i,j,:].mean(axis=1).shape)
            count += 1

            
    
    plt.subplots_adjust(hspace=-0.5,wspace=0.2)
    fig.supxlabel("Frequency (kHz)", fontsize=6)
    fig.supylabel("Sound level (dB SPL)", fontsize=6)

    return fig, axs

def FRAimg(pathToFRA, xlabels = [4,8,16,32,64], ylabels = [90,70,50,30], vmin=0, 
           vmax = 0.12, tickfontsize = 6, labelfontsize = 6):
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams["font.family"] = "Arial"
    img = mpimg.imread(pathToFRA)
    fig,ax = plt.subplots(figsize=(1.6,1.5), layout="tight")
    ax.imshow(img, cmap='Greys',vmin=0,vmax=vmax)
    ax.tick_params(axis='both',length=2, pad=1)
    ax.tick_params(axis='x',length=2, pad=1)    
    ax.set_xticks(np.arange(0,5)*330+330/2, xlabels, fontsize=tickfontsize)
    ax.set_xlabel("Frequency (kHz)", fontsize=labelfontsize, labelpad=0)
    ax.set_ylabel("Sound level (dB SPL)", fontsize=labelfontsize, labelpad=1)
    ax.set_yticks(np.arange(0,4)*330+330/2, ylabels, fontsize=tickfontsize)

    return fig, ax

def return_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
            return "**"
    elif p < 0.05:
        return "*"
    else:
        return None
    
def figQuality(fig, ax):
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rc('axes', labelsize=6) 
    plt.rc('xtick', labelsize=6) 
    plt.rc('ytick', labelsize=6) 
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = "Arial"
    ax.tick_params(axis='both', which='major', length=2, pad = 2)
    ax.spines[['right', 'top']].set_visible(False)

    return fig, ax
        