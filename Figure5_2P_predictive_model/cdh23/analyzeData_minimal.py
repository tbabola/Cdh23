from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib import animation 
from matplotlib.animation import FFMpegFileWriter
import cdh23.animator
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from scipy import stats
import itertools
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class analyzeTheData:

    #data is 200 trials for one mouse
    def __init__(self, data, stim, mouse, out_dir) :
        self.mouse = mouse
        self.directory = out_dir
        #set output dir accordingly
        self.output_dir = Path("/Volumes/Crucial X8/Cdh23/Sean/new_outputs", self.directory, self.mouse)
        isExist = os.path.exists(self.output_dir)
        if not isExist:
        # Create a new directory because it does not exist
            os.makedirs(self.output_dir)
        self.data = data
        self.frames_pre_stim = 29 #was 14
        self.frames_post_stim = 45
        self.start_stim = 30 #was 15
        self.end_stim = 38 #was 23
        self.trial_type   = stim[0][:, 0] #only 0 for first session, #stimHistories[mou] is 200x3, len 200
        self.trial_types  = ["4 kHz", "8 kHz", "16 kHz", "32 kHz", "64 kHz"]
        self.attenuations = ["0 dB", "20 dB", "40 dB", "60"]
        self.trials       = self.data #len 200
        self.trial_size   = self.trials[0].shape[1] #was trials[0]
        self.Nneurons     = self.trials[0].shape[0]
        # list of arrays containing the indices of trials of each type (t_type_ind[0] contains the
        # indices of trials of type trial_types[0])
        #t_type_ind = [np.argwhere(np.array(trial_type) == t_type)[:] for t_type in trial_types]
        self.t_type_ind = [range(40),range(40,80), range(80, 120), range(120, 160), range(160,200)]
        self.a_type_ind = [range(10),range(10,20), range(20, 30), range(30, 40)]


        self.shade_alpha      = 0.2
        self.lines_alpha      = 0.8
        #self.pal = sns.color_palette('husl', 9)
        self.pal = [(44/256, 123/256, 182/256), (171/256, 217/256, 233/256), (255/256, 255/256, 191/256), (253/256, 174/256, 97/256), (215/256, 25/256, 28/256)]
        #%config InlineBackend.figure_format = 'svg'

    #Function concatenates trial matrices as a tensor and z scores 
    def getZScoredConcat(self) :
        Xr = np.vstack([t[:, self.frames_pre_stim:self.frames_post_stim].mean(axis=1) for t in self.trials]).T
        Xr_sc = self.z_score(Xr)
        return Xr_sc

    def z_score(self, X):
        # X: ndarray, shape (n_features, n_samples)
        ss = StandardScaler(with_mean=True, with_std=True)
        Xz = ss.fit_transform(X.T).T
        return Xz

    def kFolds_classify(self, Xr_sc, folds) :

        # Initialize KFold cross-validation
        kf = StratifiedKFold(n_splits=folds)
        
        #initialize and load ground truth labels
        ytrue = np.zeros((200))
        counter = 0
        indCount = 0
        for t, ind in enumerate(self.t_type_ind):
            for k in range(4) :
                for rep in range(10) :
                    ytrue[indCount] = counter
                    indCount = indCount + 1
                counter = counter + 1

        #shuffle ground truth labels
        shuffled_indices = np.random.permutation(len(ytrue))
        shuffled = ytrue[shuffled_indices]

        # Lists to store the accuracy scores for each fold
        accuracies = []
        accuracies_chance = []
        cw_accuracies = []
        cw_chance = []
        confusion_matrices = []

        pca = PCA()
        pca.fit_transform(Xr_sc.T) #this fit is not used to train, just to determine num PCs by cummulative variance 
        sum = 0
        found = False
        for i in range(len(pca.explained_variance_ratio_)):
            sum = sum + pca.explained_variance_ratio_[i]
            #print(sum)
            if sum >= 0.90 and found == False : #PCs explaining 90% of the variance
                pcsNeeded = i
                found = True

        # Loop through each fold
        fold = 1
        for train_index, test_index in kf.split(Xr_sc.T, ytrue):

            # Split the data into training and testing sets for this fold
            y_train, y_test = ytrue[train_index], ytrue[test_index]
            shuffled_train, shuffled_test = shuffled[train_index], shuffled[test_index]

            #pca data
            #this block prevents info leakage
            Xr_train, Xr_test = Xr_sc[:, train_index], Xr_sc[:, test_index]
            pca = PCA(n_components=pcsNeeded)
            pc_train = pca.fit_transform(Xr_train.T)
            pc_test = pca.transform(Xr_test.T)

            lda_pcs = LinearDiscriminantAnalysis()
            lda_pcs.fit(pc_train, y_train)

            chance_classifier =  LinearDiscriminantAnalysis()
            chance_classifier.fit(pc_train, shuffled_train)

            # Make predictions on the test set
            y_pred = lda_pcs.predict(pc_test)
            chance_pred = chance_classifier.predict(pc_test)

            # Calculate accuracy: lda
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            # Calculate accuracy: chance lda classifier
            accuracy_chance = accuracy_score(y_test, chance_pred)
            accuracies_chance.append(accuracy_chance)

            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices.append(cm)
            cmChance = confusion_matrix(y_test, chance_pred)
            
            class_wise_accuracy = cm.diagonal() / cm.sum(axis=1)
            class_wise_chance = cmChance.diagonal() / cmChance.sum(axis=1)
            cw_accuracies.append(class_wise_accuracy)
            cw_chance.append(class_wise_chance)

        # Calculate the mean accuracy across all folds
        mean_accuracy = np.mean(accuracies)
        mean_accuracy_chance = np.mean(accuracies_chance)

        # Print and return the mean accuracy
        print(f"Accuracy- LDA fit on PCA: {mean_accuracy:.2f}")
        print(f"Chance Accuracy- LDA fit on PCA with shuffled labels: {mean_accuracy_chance:.2f}")
        print("Classwise accuracy")
        return mean_accuracy, mean_accuracy_chance, cw_accuracies, cw_chance, confusion_matrices #mean k fold accuracy, mean k fold chance accuracy, classwise accuracies for all folds, chance classwise accuracies for all folds
       

