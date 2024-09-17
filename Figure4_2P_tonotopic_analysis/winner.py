import pandas as pd
import numpy as np
import scipy

def get_winners():
    pass

def winner_take_all(data, radius = 50):
    winner_take_all_radius = radius

    mouse_str = []
    images = []

    for mouse in data['Mouse'].unique():
        temp_df = data[data['Mouse']==mouse]
        image = np.zeros((1024,1024,5))
        for charFreq in temp_df.sort_values('charFreq')['charFreq'].unique():
            temp_df2 = temp_df[temp_df['charFreq']==charFreq]
            for neuron in temp_df2['neuron'].unique():
                temp_df3 = temp_df2[temp_df2['neuron']==neuron][['med','charFreq_amp']]
                cell_med = temp_df3['med'].values[0]
                image[cell_med[1],cell_med[0],charFreq] = temp_df3['charFreq_amp'].values[0]
        image = scipy.ndimage.gaussian_filter(image,sigma=(winner_take_all_radius, winner_take_all_radius, 0))
        no_value_idc = image.mean(axis=-1)==0
        image = np.nanargmax(image,axis=-1)
        image[no_value_idc] = -1
        mouse_str.append(mouse)
        images.append(image)  
        
    return pd.DataFrame({"Mouse":mouse_str,"winner_image":images})