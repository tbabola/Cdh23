import alphashape
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plotCharFreqMap(data, mouse, alpha = 0.1, figsize = (1.5,1.5), showIntensity = False):
    fig, ax = plt.subplots(figsize = figsize)
    ex = data[data['Mouse']==mouse]
    colors_ito = {'black':[0,0,0], 'orange':[230,159,0], 'skyblue':[86,180,233], 'green':[0,158,115],
        'yellow':[240,228,66], 'blue':[0,114,178], 'darkorange':[213,94,0], 'purple': [204,121,167]}
    colors_ito = {key: (np.array(colors_ito[key])/255).tolist() for key in colors_ito.keys()}

    colors = {0:colors_ito['blue'],1:colors_ito['skyblue'],2:colors_ito['green'], 3:colors_ito['yellow'], 4:colors_ito["darkorange"]}

    cmap_dict = {}
    for i, color in enumerate(['red', 'green', 'blue']):
        cmap_dict[color] = [colors[ex][i]/255 for ex in colors.keys()]
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cmap_dict, N=5)

    test = np.zeros((1024,1024,3))
    test[:] = 1
    #test[:]=1
    #test[:,:,3]=1

    plt.imshow(test,vmin=-1,vmax=1)
    for cf in ex.sort_values(by='charFreq')['charFreq'].unique():
        temp = ex[(ex['charFreq']==cf)]
        for i, cell in temp.iterrows():
            points = np.stack(cell[['xpix', 'ypix']]).T
            #print(points.shape, points.tolist())
            ashape = alphashape.alphashape([tuple(point) for point in points], alpha)
            vertices = np.array(ashape.exterior.coords)
            codes = []
            for i, vertex in enumerate(vertices):
                if i == 0:
                    codes.append(Path.MOVETO)
                elif i == len(vertices)-1:
                    codes.append(Path.CLOSEPOLY)
                else:
                    codes.append(Path.LINETO)

            if showIntensity:
                pathpatch = patches.PathPatch(Path(vertices, codes), edgecolor=None, facecolor = colors[cf], linewidth=0, alpha = cell['log10_cfamp_norm'])
            else:
                pathpatch = patches.PathPatch(Path(vertices, codes), edgecolor=None, facecolor = colors[cf], linewidth=0)

            ax.add_patch(pathpatch)
            edgepatch = patches.PathPatch(Path(vertices, codes), edgecolor=[0.8,0.8,0.8], alpha= 1,facecolor = None, fill=False,linewidth=0.1)
            ax.add_patch(edgepatch)
            # ax.add_patch(PolygonPatch(ashape, edgecolor=None, facecolor = colors[cf], linewidth=0, alpha = cell['log10_cfamp_norm']))
            # patch = PolygonPatch(ashape, edgecolor=None, facecolor = colors[cf], linewidth=0, alpha = cell['log10_cfamp_norm'])
            # ax.add_patch(PolygonPatch(ashape, edgecolor=[0.8,0.8,0.8], alpha= 1,facecolor = None, fill=False,linewidth=0.25))
            #break
    ax.axis('off')
    
    return fig, ax