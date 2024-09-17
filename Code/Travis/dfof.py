import numpy as np
import scipy
import mat73
import time

class processFluor():
    def __init__(self, dir = None, F = None) -> None:
        self.dir = dir
        self.F = F

    def __loadSoundFile__(self):
        soundfile = list(self.dir.glob("sound_file*.mat"))
        if not soundfile:
            print("No sound file found.")
            return None
        elif len(soundfile) > 1:
            print("Multiple sound files fou nd.")
            return None
        else: 
            print("Sound file found.")
            return mat73.loadmat(soundfile[0])

    def loadTraces(self, neuCorrection = 0.7):
        suite2p_dir = self.dir / "suite2p/plane0/"
        if suite2p_dir.exists():
            iscell = np.load(suite2p_dir / "iscell.npy", allow_pickle=True)
            F = np.load(suite2p_dir / "F.npy",allow_pickle=True)
            Fneu = np.load(suite2p_dir / "Fneu.npy",allow_pickle=True)
                           
            iscell = iscell.astype(bool)[:,0]
            F = F[iscell,:]
            Fneu = Fneu[iscell,:]
            F = F - neuCorrection*Fneu
            self.F = F
        else:
            print("No suite2p files found.")

    def calc_dfof(self, window = 100, step = 30, percentile = 10, cutoff = None):
        def baseline(F):
            ##F: an m x t array of fluorescence values
            #calculate baseline based on rolling window quantile
            temp = F.copy()
                
            temp = np.pad(temp,((0,0),(window-step,window-step)), mode='edge')
            length = temp.shape[1]
            
            start =time.time()
            times = []
            baseline_vals = []
            for i in np.arange(0,length-window+1,step):
                times.append(i)
                baseline_vals.append(np.percentile(temp[:,i:i+window], percentile, axis=1))

            times = np.stack(times)
            
            baseline_vals = np.stack(baseline_vals)
            print(time.time()-start, " is the time for calculating percentile")

            start = time.time()
            #interpolate values so it is the same shape as the input flourescence
            start = time.time()
            f = scipy.interpolate.interp1d(times, baseline_vals, axis=0)
            bl = f(np.arange(times[0],F.shape[1])).T
            print(time.time()-start, "is the time for interpolating.")

            return bl

        F = self.F
        start = time.time()
        bl = baseline(F)
        print(time.time()-start, " to calculate baseline")
        
        if cutoff:
            #cutoff baseline values so that really small values of bl don't cause dfof to explode (i.e. divide by ~0)
            bl[(bl > 0) & (bl < cutoff)] = cutoff
            bl[(bl<= 0) & (bl > -cutoff)] = -cutoff
        
        dfof = np.divide(F-bl,np.abs(bl))
        self.dfof = dfof
        self.bl = bl
        return (dfof, bl)
    
    def filter_dfof(self, filter_fs = 15, filter_w = 2):
        ###TO DO
        #set up low pass filter for smoothed signal
        dfof = self.dfof
        b, a = scipy.signal.butter(2, filter_w, 'low', fs=filter_fs)
        dfof_sm = scipy.signal.filtfilt(b,a,dfof)
        self.dfof_sm = dfof_sm
        return None
    
    def to_mat(self, save_path = None, save_dfof = True, save_dfof_sm = True, save_bl=True):
        if not save_path:
            save_path = self.dir / "analysis"
            save_path.mkdir(exist_ok=True)
        print("Files will be saved to ",str(save_path))
        data = {}
        if save_dfof:
            data['dfof']= self.dfof
        if save_dfof_sm:
            data['dfof_sm']= self.dfof_sm
        if save_bl:
            data['dfof_bl']= self.bl
        
        scipy.io.savemat(save_path / "traces.mat", data)

    def unmix(self, numBaseline = 5, framesAfter = 15, divideFrame = 1):
        start = time.time()
        self.soundData = self.__loadSoundFile__()

        if self.soundData is not None:
            self.nStims = len(set(self.soundData['stimPlayedID']))
            self.nAttens = self.soundData['atten'][0].size
            soundData = self.soundData #for ease of typing 

            frames = self.soundData['stimOnFrame']-1 #zero indexing in python
            attens = np.int32(soundData['atten'][0])
            if np.isscalar(attens):
                attens = np.array(attens)[np.newaxis]
                print(attens.shape)

            freqID = np.unique(soundData['stimPlayedID'])
            nFreqs = len(freqID)
            nRep = np.int32(soundData['nRep'])
            framesToAnalyze = numBaseline + framesAfter

            self.unmixed = np.zeros(shape=(self.F.shape[0], framesToAnalyze,
                                                nFreqs,self.nAttens,np.int32(soundData['nRep'])))#,dtype=np.uint16)

            for freq in range(nFreqs):
                for atten in range(self.nAttens):
                    index = np.logical_and(soundData['stimPlayedID']==freqID[freq],soundData['stimPlayedAtten']==attens[atten])
                    framesForStimAndLevel = frames[index]//divideFrame
                    #print(framesForStimAndLevel)
                    startFrames = np.int32(framesForStimAndLevel - numBaseline)
                    for repeat in range(nRep):
                        self.unmixed[:,:,freq,atten,repeat] = self.dfof[:,startFrames[repeat]:startFrames[repeat]+framesToAnalyze]

            print("{} seconds to unmix file".format(time.time()-start))
        else:
            print("No sound file, so unmixing cannot be performed")