from pathlib import Path
import mat73
from PIL import Image
from skimage import io, filters
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import pickle
import pandas as pd
import datetime
import scipy

class groupData():
    def __init__(self, dir = None, analysis_file_name = "wfstack.pkl"):
        self.homedir = dir
        self.mice = self.loadMouseInfo(analysis_file=analysis_file_name)
    
    def loadMouseInfo(self, analysis_file = "wfstack.pkl"):
        if (self.homedir / "MouseInfo.csv").is_file():
            home = self.homedir
            mice = pd.read_csv(home / "MouseInfo.csv")
            mice['data'] = object
            mice['path'] = Path()
            mice['data_6mo'] = object
            mice['path_6mo'] = Path()

            for i, mouse in mice.iterrows():
                homedir = home / mouse['Mouse'] / "Widefield"
                DOB = mouse['DOB']
                DOB = datetime.datetime.strptime(DOB,"%m/%d/%Y") 
                path = list(homedir.glob('[0-9]' * 6))[0]
                data = path / analysis_file
                if data.exists(): 
                    file = open(str(data),'rb')
                    object_file = pickle.load(file)
                    mice.loc[i,'data'] = object_file
                    mice.loc[i,'path'] = path
                    file.close()
                else:
                    print(mouse, " has no data")
                if len(list(homedir.glob('[0-9]' * 6))) > 1:
                    for path2 in mice.loc[i,'path'].parent.glob(6*"[0-9]"):
                        date = datetime.datetime.strptime(str(path2.name),"%y%m%d")
                        if (date-DOB).days > 180:
                            if (path2 / analysis_file).exists():
                                data = path2 / analysis_file
                                file = open(str(data),'rb')
                                object_file = pickle.load(file)
                                mice.loc[i,'data_6mo'] = object_file
                                mice.loc[i,'path_6mo'] = path2
                                file.close()
                                break
                            else:
                                print(mouse, "does not have 6 months data")

            return mice
        else:
            print("Directory initialized does not have MouseInfo.csv")
            return None



class widefieldStack():
    def __init__(self, dir = None, numBaseline = 30, startResponse = 40, endResponse = 45, divideFrame = 1) -> None:
        self.dir = dir
        self.numBaseline = numBaseline
        self.startResponse = startResponse
        self.endResponse = endResponse
        self.percentileResponse = None
        self.percentile = None
        self.soundData = self.__loadSoundFile__()
        self.unmixed = None
        self.unmixed_dfof = None
        self.s2p = (self.dir / "suite2p").is_dir()
        self.stack = None
        self.FRA = None
        self.area = None
        self.divideFrame = divideFrame #variable for holding a divisor for frame counts (averaging frames)
        self.anomolies = None
        self.thresholds = None
        if self.soundData is not None:
            self.nStims = len(set(self.soundData['stimPlayedID']))
            self.nAttens = self.soundData['atten'][0].size

    def run_pipeline(self, save_dir="maps3"):
        print(self.dir)
        #self.stack = self.__loadStack__()
        self.stack = imresize(self.__loadStack__(),np.array([100,100]))
        self.unmix()
        self.removeAnamolies()
        self.generateMap(frameToTake=45, save_dir=save_dir)
        self.generateFRA(startFrame=40, endFrame=45,save_dir=save_dir)
        self.generateFRAmovie(save_dir=save_dir)
        self.calculate_percentiles(percentile=99)
        self.calculate_area(tresholdStd=1.5)
        #self.calculate_thresholds() 
        self.dumpRawData()
        import pickle
        file = open(self.dir / "wfstack_3.pkl", 'wb')
        pickle.dump(self, file)
        file.close()

    def __loadSoundFile__(self):
        soundfile = list(self.dir.glob("sound_file*.mat"))
        if not soundfile:
            print("No sound file found.")
            return None
        elif len(soundfile) > 1:
            print("Multiple sound files found.")
            return None
        else: 
            print("Sound file found.")
            return mat73.loadmat(soundfile[0])

    def __loadStack__(self):
        tiffs = self.dir.glob("*.tif")
        start = time.time()
        if self.s2p:
            print("Suite2P file loading...")
            ops = np.load(self.dir / "suite2p/plane0/ops.npy", allow_pickle=True)[()]
            width, height = ops['Lx'], ops["Ly"]
            data = np.fromfile(self.dir / "suite2p/plane0/data.bin", dtype=np.int16)
            print("{} seconds to load file".format(time.time()-start))
            return np.reshape(data,newshape=(-1,height,width))
        else:
            print("Tiffs loading...")
            data = []
            for tiff in tiffs:
                data.append(io.imread(tiff))
            print("{} seconds to load file".format(time.time()-start))
            return np.concatenate(data)

    def unmix(self, framesAfter = 100, forceReload = False, use_dfof = False):
        start = time.time()
        if self.soundData is not None:
            soundData = self.soundData #for ease of typing 

            #check for matlab file or python file, convert to python
            if ((self.dir / 'data_unmixed.mat').is_file() or (self.dir / 'widefield.pkl').is_file()) and not forceReload:
                print("data_unmixed found")
            else:
                print("we are in here")
                if self.stack is None:
                    self.stack = self.__loadStack__()

                frames = self.soundData['stimOnFrame']-1 #zero indexing in python
                attens = np.int32(soundData['atten'][0])
                if np.isscalar(attens):
                    attens = np.array(attens)[np.newaxis]
                    print(attens.shape)

                freqID = np.unique(soundData['stimPlayedID'])
                nFreqs = len(freqID)
                nRep = np.int32(soundData['nRep'])
                framesToAnalyze = self.numBaseline + framesAfter

                self.unmixed = np.zeros(shape=(framesToAnalyze,self.stack.shape[1],self.stack.shape[2],
                                                    nFreqs,self.nAttens,np.int32(soundData['nRep'])))#,dtype=np.uint16)

                for freq in range(nFreqs):
                    for atten in range(self.nAttens):
                        index = np.logical_and(soundData['stimPlayedID']==freqID[freq],soundData['stimPlayedAtten']==attens[atten])
                        framesForStimAndLevel = frames[index]//self.divideFrame
                        #print(framesForStimAndLevel)
                        startFrames = np.int32(framesForStimAndLevel - self.numBaseline)
                        for repeat in range(nRep):
                            self.unmixed[:,:,:,freq,atten,repeat] = self.stack[startFrames[repeat]:startFrames[repeat]+framesToAnalyze,:,:]
            print("{} seconds to unmix file".format(time.time()-start))
        else:
            print("No sound file, so unmixing cannot be performed")

    ##look for anomolous stacks
    def anomalies(self, std_thr = 2):
        #takes stacks and looks for individual responses that differ significantly from the mean
        unmixed = self.unmixed
        mean_bl = unmixed.mean(axis=(-1))
        nRep = np.int32(self.soundData['nRep'])
        diffs = np.square(unmixed - np.repeat(np.expand_dims(mean_bl,axis=-1),nRep,axis=-1))
        diffs_sum = np.sum(diffs,axis=(0,1,2))

        return diffs_sum > (diffs_sum.mean() + std_thr * diffs_sum.std())

    def removeAnamolies(self,  std_thr = 2):
        unmixed_nan = self.unmixed.copy()
        anomaly_index = self.anomalies(std_thr = std_thr)
        for index in np.argwhere(anomaly_index):
            unmixed_nan[:,:,:,index[0],index[1],index[2]] = np.NaN
        self.anomalies = anomaly_index
        self.unmixed = unmixed_nan

    def generateMap(self, frameToTake = 25, freqToTake = np.array([4,2,0]), save=True, save_mean=False, save_dir = "maps2", percentile_norm = 99.9, raw_multiplier = 10):
        if self.unmixed is not None:
            mean_stack = np.nanmean(self.unmixed,axis=5)
            map = np.zeros(shape=(mean_stack.shape[0],mean_stack.shape[1], mean_stack.shape[2],3,mean_stack.shape[4]))
            for i, freq in enumerate(freqToTake):
                map[:,:,:,i,:] = mean_stack[:,:,:,freq,:]
            bl = np.nanmean(map[:self.numBaseline,:,:,:,:],axis=0)
            blsubt = map - bl
            prctiles = np.percentile(blsubt[frameToTake,:,:,:,:],q=percentile_norm,axis=(0,1))
            norm = np.divide(blsubt,prctiles)
            norm[norm < 0] = 0
            norm[norm > 1] = 1

            dfof = np.divide(blsubt,bl)*raw_multiplier
            dfof[dfof < 0] = 0
            dfof[dfof > 1] = 1

            (self.dir / save_dir).mkdir(exist_ok=True)

            for i in range(dfof.shape[-1]):
                io.imsave(self.dir / save_dir / (str(i) + ".png"), np.uint8(filters.gaussian(norm[frameToTake,:,:,:,i],0.5)*255))
                io.imsave(self.dir / save_dir / (str(i) + "_raw.png"), np.uint8(filters.gaussian(dfof[frameToTake,:,:,:,i],0.5)*255))

            if save_mean:
                io.imsave(self.dir / save_dir / ("mean.tif"), np.mean(self.stack,axis=0))
        else:
            print("No unmixed file found, so can't generate maps.")


    def generateFRA(self, startFrame = 20, endFrame = 25, vmax= 0.15, cmap = 'binary', save=True, save_dir = "maps2"):
        averages = []
        numBaseline = 10

        for freq in range(self.nStims):
            concat_atten = []
            for atten in range(self.nAttens):
                mean_freq_atten = np.nanmean(self.unmixed[:,:,:,freq,atten,:],axis=-1)
                Fo = mean_freq_atten[0:numBaseline,].mean(axis=0)
                dfof = np.nanmean(np.divide(mean_freq_atten-Fo,Fo)[startFrame:endFrame,:],axis=0)
                if atten == 0:
                    concat_atten = dfof
                else:
                    concat_atten = np.concatenate((concat_atten,dfof),axis=0)
            if freq == 0:
                wholecat = concat_atten
            else:
                wholecat = np.concatenate((wholecat,concat_atten),axis=1)

        self.FRA = wholecat
        
        if save:
            (self.dir / save_dir).mkdir(exist_ok=True)
            plt.imsave(self.dir / save_dir / "FRA.png", wholecat, vmin = 0, vmax = vmax, cmap=cmap)

    def generateFRAmovie(self, save_dir = "maps2", filename = "FRA.mp4", max_norm = 0.1, mode = "dfof"):
        if self.unmixed_dfof is None:
            if mode == "dfof":
                self.dFoF(self.unmixed)
                stack = self.unmixed_dfof
            elif mode == "df":
                self.dF(self.unmixed)
                stack = self.unmixed_dfof
            elif mode == "raw":
                stack = self.unmixed
        
        self.max_norm = max_norm

        for freq in range(self.nStims):
            concat_atten = []
            for atten in range(self.nAttens):
                mean_freq_atten = np.nanmean(stack[:,:,:,freq,atten,:],axis=-1)
                if atten == 0:
                    concat_atten = mean_freq_atten
                else:
                    concat_atten = np.concatenate((concat_atten,mean_freq_atten),axis=1)

            if freq == 0:
                wholecat = concat_atten
            else:
                wholecat = np.concatenate((wholecat,concat_atten),axis=2)
        
        size = wholecat.shape
        wholecat[wholecat < 0] = 0
        wholecat[wholecat > max_norm] = max_norm
        wholecat *= 255/max_norm
        wholecat = wholecat.astype(np.uint8)

        out = cv2.VideoWriter(str(self.dir / save_dir / filename), cv2.VideoWriter_fourcc(*'mp4v'), 30, (size[2], size[1]), isColor = False)
        for i in range(wholecat.shape[0]):
            data = wholecat[i,:]
            out.write(data)
        out.release()
        del wholecat

    def calculate_area(self, tresholdStd = 3):
        if self.unmixed_dfof is None:
            self.dFoF(self.unmixed)
        
        stack_dfof = self.unmixed_dfof

        #area measurement
        dfof_mean = np.nanmean(stack_dfof[:self.numBaseline,:],axis=0)
        dfof_std = np.nanstd(stack_dfof[:self.numBaseline,:],axis=0)
        threshold = dfof_mean + tresholdStd *dfof_std
        signal = stack_dfof > threshold
        signal = np.nanmean(signal,axis=(1,2))
        self.area = signal

    def calculate_percentiles(self, percentile = 95):
        if self.unmixed_dfof is None:
            self.dFoF(self.unmixed)
        
        stack_dfof = self.unmixed_dfof

        stack_dfof_response = np.nanmean(stack_dfof[self.startResponse:self.endResponse,:],axis=-1)
        stack_dfof_response = np.max(stack_dfof_response,axis=0)

        percentiles = np.percentile(np.abs(stack_dfof_response),q=percentile,axis=(0,1))
        mask = np.abs(stack_dfof_response) > percentiles
        mask = mask.astype(float)
        mask[mask==False] = np.NaN
        percentileResponse = stack_dfof * np.repeat(np.expand_dims(mask,axis=-1),repeats=stack_dfof.shape[-1], axis=-1)
        percentileResponse = np.nanmean(percentileResponse,axis=(1,2))
        self.percentileResponse = percentileResponse

    def calculate_thresholds(self,startResponse=44, endResponse=46, p_crit = 0.05):
        threshold = {key: -1 for key in range(self.nStims)}
        for freq in range(self.nStims):
            for atten in range(self.nAttens):
                baseline = self.percentileResponse[:self.numBaseline,freq,atten].mean(axis=0)
                stimulus = self.percentileResponse[startResponse:endResponse,freq,atten].mean(axis=0)
                p = scipy.stats.ttest_rel(baseline,stimulus,nan_policy="omit",alternative="less").pvalue
                if p > 0.5:
                    break
                elif p < p_crit:
                    threshold[freq] = self.soundData['atten'][0][atten]
        self.thresholds = threshold

    def dF(self, array):
        Fo = np.nanmean(array[:self.numBaseline,:],axis=0)
        self.unmixed_dfof = array-Fo

    def dFoF(self, array):
        Fo = np.nanmean(array[:self.numBaseline,:],axis=0)
        self.unmixed_dfof = np.divide(array-Fo,Fo)
        
    def dumpRawData(self):
        self.unmixed = None
        self.stack = None
        self.unmixed_dfof = None


def findUnanalyzedStacks(basedir, pattern = "**/Widefield/[0-9]*", foldername = "maps"):
    nomapsdir = []
    print(pattern, foldername)
    for dir in basedir.glob(pattern):
        if not len(list(dir.glob(foldername))):
            nomapsdir.append(dir)

    return nomapsdir

def plotFreqAtten(array, ymin = -0.01, ymax=0.01):
    numcols = array.shape[1]
    numrows = array.shape[2]
    plt.subplot()
    count = 1
    
    for j in range(numrows):
        for i in range(numcols):
            plt.subplot(numrows,numcols,count)
            plt.plot(array[:,i,j])
            count += 1
            plt.ylim((ymin,ymax))

def plotFreqAttenMulti(array, ymin = -0.01, ymax=0.01, xmin = 0, xmax = 100):
    numcols = array.shape[1]
    numrows = array.shape[2]
    numanimals = array.shape[3]
    plt.subplot()
    
    for k in range(numanimals):
        count = 1
        for j in range(numrows):
            for i in range(numcols):
                plt.subplot(numrows,numcols,count)
                plt.plot(array[:,i,j,k],c=[0.9,0.9,0.9])
                count += 1
                plt.xlim((xmin,xmax))
                plt.ylim((ymin,ymax))

        count = 1
        for j in range(numrows):
            for i in range(numcols):
                plt.subplot(numrows,numcols,count)
                plt.plot(np.median(array[:,i,j,:],axis=1),c=[0,0,0])
                count += 1

def plotFreqAttenMulti1(array, ymin = -0.01, ymax=0.01, xmin = 0, xmax = 100):
    numcols = array.shape[1]
    numrows = array.shape[2]
    numanimals = array.shape[3]
    plt.subplot()
    
    for k in range(numanimals):
        count = 1
        for j in range(numrows):
            for i in range(numcols):
                plt.subplot(numrows,numcols,count)
                plt.plot(array[:,i,j,k],c=[0.9,0.9,0.9])
                count += 1
                plt.xlim((xmin,xmax))
                plt.ylim((ymin,ymax))
                plt.subplots_adjust(hspace=0,wspace=0)

        count = 1
        for j in range(numrows):
            for i in range(numcols):
                plt.subplot(numrows,numcols,count)
                plt.plot(np.nanmean(array[:,i,j,:],axis=-1),c=[0,0,0])
                count += 1
                plt.subplots_adjust(hspace=0)

def imresize(img_stack,shape):
    import cv2
    width = shape[0]
    height = shape[1]
    img_stack_sm = np.zeros((img_stack.shape[0], width, height))

    for idx in range(img_stack.shape[0]):
        img = img_stack[idx, :, :]
        img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img_stack_sm[idx, :, :] = img_sm
    
    return img_stack_sm

class wfstack2():
    def __init__(self, dir, numBaseline = 30, startResponse = 40, endResponse = 45):
        self.dir = dir
        self.numBaseline = numBaseline
        self.startResponse = startResponse
        self.endResponse = endResponse
        self.percentileResponse = None
        self.soundData = self.__loadSoundFile__()
        self.unmixed = None
        self.s2p = (self.dir / "suite2p").is_dir()
        self.stack = None
        self.FRA = None
        self.area = None
        if self.soundData is not None:
            self.nStims = len(set(self.soundData['stimPlayedID']))
            self.nAttens = self.soundData['atten'][0].size

    def __loadStack__(self):
        tiffs = self.dir.glob("*.tif")
        start = time.time()
        if self.s2p:
            print("Suite2P file loading...")
            im = tifffile.imread(list(tiffs)[0], key=0)
            print(im.shape)
            width, height = im.shape[0], im.shape[1]
            data = np.fromfile(self.dir / "suite2p/plane0/data.bin", dtype=np.int16)
            print("{} seconds to load file".format(time.time()-start))
            return np.reshape(data,newshape=(-1,width,height))
        else:
            print("Tiffs loading...")
            data = []
            for tiff in tiffs:
                data.append(tifffile.imread(tiff))
            print("{} seconds to load file".format(time.time()-start))
            return np.concatenate(data)

    def __loadSoundFile__(self):
        soundfile = list(self.dir.glob("sound_file*.mat"))
        if not soundfile:
            print("No sound file found.")
            return None
        elif len(soundfile) > 1:
            print("Multiple sound files found.")
            return None
        else: 
            print("Sound file found.")
            return mat73.loadmat(soundfile[0])
        
    def run_pipeline(self, save_dir="maps5"):
        print(self.dir)
        self.stack = imresize(self.__loadStack__(),np.array([100,100]))
        length = self.stack.shape[0]
        (dfof, dfof_sm, baseline) = self.dfof(np.reshape(self.stack,(length,-1)).T, window= 3000, step = 1000, quantile=50, filter_fs = 30)
        self.stack = dfof_sm.T.reshape((length,100,100))
        self.unmix()
        self.generateMap(frameToTake=45, save_dir=save_dir)
        self.generateFRA(startFrame=40, endFrame=45,save_dir=save_dir)
        self.generateFRAmovie(save_dir=save_dir)
        self.calculate_percentiles()
        self.calculate_area(thresholdStd=1.5)

        del self.stack
        import pickle
        file = open(self.dir / "wfstack_2.pkl", 'wb')
        pickle.dump(self, file)
        file.close()
        
        
    def unmix(self, framesAfter = 100, forceReload = False, use_dfof = False):
            start = time.time()
            if self.soundData is not None:
                soundData = self.soundData #for ease of typing 

                #check for matlab file or python file, convert to python
                if ((self.dir / 'data_unmixed.mat').is_file() or (self.dir / 'widefield.pkl').is_file()) and not forceReload:
                    print("data_unmixed found")
                else:
                    print("we are in here")
                    if self.stack is None:
                        self.stack = self.__loadStack__()

                    frames = self.soundData['stimOnFrame']-1 #zero indexing in python
                    attens = np.int32(soundData['atten'][0])
                    if np.isscalar(attens):
                        attens = np.array(attens)[np.newaxis]
                        print(attens.shape)

                    freqID = np.unique(soundData['stimPlayedID'])
                    nFreqs = len(freqID)
                    nRep = np.int32(soundData['nRep'])
                    framesToAnalyze = self.numBaseline +framesAfter

                    self.unmixed = np.zeros(shape=(framesToAnalyze,self.stack.shape[1],self.stack.shape[2],
                                                        nFreqs,self.nAttens,np.int32(soundData['nRep'])))#,dtype=np.uint16)

                    for freq in range(nFreqs):
                        for atten in range(self.nAttens):
                            index = np.logical_and(soundData['stimPlayedID']==freqID[freq],soundData['stimPlayedAtten']==attens[atten])
                            framesForStimAndLevel = frames[index]

                            startFrames = np.int32(framesForStimAndLevel - self.numBaseline)
                            for repeat in range(nRep):
                                self.unmixed[:,:,:,freq,atten,repeat] = self.stack[startFrames[repeat]:startFrames[repeat]+framesToAnalyze,:,:]
                print("{} seconds to unmix file".format(time.time()-start))
            else:
                print("No sound file, so unmixing cannot be performed")

    def generateMap(self, frameToTake = 25, save=True, save_dir = "maps3"):
        if self.unmixed is not None:
            mean_stack = np.mean(self.unmixed,axis=5)
            map = mean_stack[:,:,:,4::-2,:]
            dfof = map*10
            dfof[dfof < 0] = 0
            dfof[dfof > 1] = 1
            (self.dir / save_dir).mkdir(exist_ok=True)

            for i in range(dfof.shape[-1]):
                print(self.dir / save_dir / (str(i) + ".png"))
              #io.imsave(self.dir / save_dir / (str(i) + ".png"), np.uint8(filters.gaussian_filter(norm[frameToTake,:,:,:,i],0.5)*255))
                io.imsave(self.dir / save_dir / (str(i) + "_raw.png"), np.uint8(filters.gaussian_filter(dfof[frameToTake,:,:,:,i],0.5)*255))
        else:
            print("No unmixed file found, so can't generate maps.")

    def generateFRA(self, startFrame = 20, endFrame = 25,save=True, save_dir = "maps2"):
        averages = []
        numBaseline = 10

        for freq in range(self.nStims):
            concat_atten = []
            for atten in range(self.nAttens):
                mean_freq_atten = np.mean(self.unmixed[:,:,:,freq,atten,:],axis=-1)
                dfof = mean_freq_atten[startFrame:endFrame,:].mean(axis=0)
                if atten == 0:
                    concat_atten = dfof
                else:
                    concat_atten = np.concatenate((concat_atten,dfof),axis=0)
            if freq == 0:
                wholecat = concat_atten
            else:
                wholecat = np.concatenate((wholecat,concat_atten),axis=1)

        self.FRA = wholecat
        
        if save:
            (self.dir / save_dir).mkdir(exist_ok=True)
            print("we here")
            plt.imsave(self.dir / save_dir / "FRA.png", wholecat, vmin = 0, vmax = 0.15, cmap='binary')

    def generateFRAmovie(self, save_dir = "maps2", filename = "FRA.mp4", max_norm = 0.1):
        
        self.max_norm = max_norm
        stack_dfof = self.unmixed

        for freq in range(self.nStims):
            concat_atten = []
            for atten in range(self.nAttens):
                mean_freq_atten = np.mean(stack_dfof[:,:,:,freq,atten,:],axis=-1)
                if atten == 0:
                    concat_atten = mean_freq_atten
                else:
                    concat_atten = np.concatenate((concat_atten,mean_freq_atten),axis=1)

            if freq == 0:
                wholecat = concat_atten
            else:
                wholecat = np.concatenate((wholecat,concat_atten),axis=2)
        
        size = wholecat.shape
        wholecat[wholecat < 0] = 0
        wholecat[wholecat > max_norm] = max_norm
        wholecat *= 255/max_norm
        wholecat = wholecat.astype(np.uint8)

        out = cv2.VideoWriter(str(self.dir / save_dir / filename), cv2.VideoWriter_fourcc(*'mp4v'), 30, (size[2], size[1]), isColor = False)
        for i in range(wholecat.shape[0]):
            data = wholecat[i,:]
            out.write(data)
        out.release()
        del wholecat

    def soundData_to_pandas(self):
        import pandas as pd
        soundData = pd.DataFrame(data=({'stimID': self.soundData['stimPlayedID'], 'stimAtten':self.soundData['stimPlayedAtten'], 'stimOnFrame':self.soundData['stimOnFrame'], 'trialNumber': 0, 'useTrial':False}))
        soundData = soundData.astype({'stimID':np.int16,'stimAtten':np.int16,'stimOnFrame':np.int16})

        for stim in soundData.stimID.unique():
            for atten in soundData.stimAtten.unique():
                soundData.loc[(soundData['stimID']==stim) & (soundData['stimAtten']==atten),'trialNumber']=np.arange(self.soundData['nRep'].astype(np.int16))
        
        return soundData
        
    def calculate_percentiles(self, percentiles = [90,95,99]):
        startResponse = 40
        endResponse = 55

        soundData = self.soundData_to_pandas()
        response = {}
        percentilesResponse = np.zeros(shape=(130,5,4,10))
        for percentile in percentiles:
            for freqID, freq in enumerate(sorted(soundData.stimID.unique())):
                for attenID, atten in enumerate(sorted(soundData.stimAtten.unique())):
                    mask = (soundData['stimID']==freq) & (soundData['stimAtten']==atten)   
                    mean_freq_atten = np.mean(self.unmixed[startResponse:endResponse,:,:,freqID, attenID,:], axis=(0,-1))
                    percentiles = np.percentile(np.abs(mean_freq_atten),q=percentile,axis=(0,1))
                    mask2 = np.abs(mean_freq_atten)>percentiles
                    totalPixels = mask2.shape[0]*mask2.shape[1]*(100-percentile)/100
                    percentilesResponse[:,freqID,attenID] = (self.unmixed[:,:,:,freqID, attenID,:] * np.repeat(mask2[:,:,np.newaxis],repeats=10, axis=-1)).sum(axis=(1,2))/totalPixels
            response[percentile]=percentilesResponse
        
        self.percentileResponse = response
    
    def calculate_area(self, thresholdStd = 2):
        stack_dfof = self.unmixed
        #area measurement
        dfof_mean = stack_dfof[:self.numBaseline,:].mean(axis=(0,3,4,5))
        dfof_std = stack_dfof[:self.numBaseline,:].std(axis=(0,3,4,5))
        threshold = dfof_mean + thresholdStd *dfof_std
        signal = stack_dfof > threshold[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis]
        self.area = signal
    

    def dfof(self, F, window = 27500, step = 13750, quantile = 10, filter_fs = 15, filter_w = 2):
        def baseline(F):
            #calculate baseline based on rolling window quantile
            temp = F.copy()
            
            #blank out the frames where a stimulus occured
            for frame in self.soundData['stimOnFrame']:
                frame = frame.astype(np.uint16)
                temp[:,frame:frame+60] = np.NAN
                
            temp = np.pad(temp,((0,0),(window-step,window-step)), mode='constant',constant_values=np.nan)
            length = temp.shape[1]

            time = []
            baseline_vals = []
            for i in np.arange(0,length-window,step):
                time.append(i)
                baseline_vals.append(np.nanpercentile(temp[:,i:i+window], quantile, axis=1))

            time = np.stack(time)
            baseline_vals = np.stack(baseline_vals)

            #interpolate values so it is the same shape as the input flourescence
            baseline = []
            for i in range(baseline_vals.shape[1]):
                baseline.append(np.interp(range(F.shape[1]), time, baseline_vals[:,i]))

            return np.stack(baseline)

        bl = baseline(F)

        #cutoff baseline values so that really small values of bl don't cause dfof to explode (i.e. divide by ~0)
        bl[(bl > 0) & (bl < 5)] = 5
        bl[(bl<= 0) & (bl > -5)] = -5
        dfof = np.divide(F-bl,np.abs(bl))

        #set up low pass filter for smoothed signal
        b, a = scipy.signal.butter(2, filter_w, 'low', fs=filter_fs)
        dfof_sm = scipy.signal.filtfilt(b,a,dfof)
        return (dfof, dfof_sm, bl)


        

    

    

