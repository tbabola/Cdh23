import numpy as np

class animater :
    def __init__(self, X, ax, trial_size, trial_types, frames_pre_stim,frames_post_stim, pal) :
        self.Xa_p = X
        self.ax = ax
        self.trial_size = trial_size
        self.trial_types = trial_types
        self.frames_pre_stim = frames_pre_stim
        self.frames_post_stim = frames_post_stim
        self.pal = pal

    def style_3d_ax(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        
    def animate(self, i):
        component_x = 0
        component_y = 1
        component_z = 2
    
        self.ax.clear() # clear up trajectories from previous iteration
        self.style_3d_ax(self.ax)
        self.ax.view_init(elev=22, azim=30)

        for t, t_type in enumerate(self.trial_types):
            x = self.Xa_p[component_x, t * self.trial_size :(t+1) * self.trial_size][0:i]
            y = self.Xa_p[component_y, t * self.trial_size :(t+1) * self.trial_size][0:i]
            z = self.Xa_p[component_z, t * self.trial_size :(t+1) * self.trial_size][0:i]

            stim_mask = ~np.logical_and(np.arange(z.shape[0]) >= self.frames_pre_stim,
                     np.arange(z.shape[0]) < (self.trial_size-self.frames_post_stim))

            z_stim = z.copy()
            z_stim[stim_mask] = np.nan
            z_prepost = z.copy()
            z_prepost[~stim_mask] = np.nan
        
            self.ax.plot(x, y, z_stim, c = self.pal[t])
            self.ax.plot(x, y, z_prepost, c=self.pal[t], ls=':')

        self.ax.set_xlim(( -50, 50))
        self.ax.set_ylim((-50, 50))
        self.ax.set_zlim((-25, 25))

        return []