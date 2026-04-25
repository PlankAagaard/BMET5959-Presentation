from aiosignal import Signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from matplotlib.patches import Polygon

#Destructive og constructive interference simulation
Destructive = False
if Destructive:
    phaseshift = np.pi
else:
    phaseshift = 0



# Leading component of the combined signal
def Combined_signal(x, polx,y, poly):
    xy_comb = x+y*np.cos(polx-poly+phaseshift)
    yx_comb = y*np.sin(polx-poly+phaseshift)
    indeces = np.abs(xy_comb)<np.abs(yx_comb)
    xy_comb[indeces] = yx_comb[indeces]
    return xy_comb


# load waveform data from file
waveform_data = np.loadtxt('waveform_data.csv', delimiter=',', skiprows=1)
x = waveform_data[:, 0]
y = waveform_data[:, 1]
polarity = waveform_data[:, 2]

N_points = len(x)
bounds = np.min(x)

# load interference data from file
interference_data = np.loadtxt('interference_data.csv', delimiter=',', skiprows=1)
Delta = interference_data[:, 0]
Is = interference_data[:, 1]


Nfinal = N_points//4



dx = 4*bounds/N_points







# plotting
fig, ax = plt.subplots(2,2, figsize=(7,20), gridspec_kw={'width_ratios': [1,1], 'height_ratios': [10, 1]})
finaly = y[-Nfinal:]
finalpol = polarity[-Nfinal:]



ax[0,0].set_aspect('equal')
ax[0,0].set_ylim(-1*10**-5,10**-5)

ax[0,0].set_xlabel('x (m)')
ax[0,0].set_ylabel('y (m)')

ax[1,1].plot(x, polarity, '.',color='purple', markersize=0.5)
ax[1,1].set_xlabel('x (m)')
ax[1,1].set_ylabel('Polarity (rad)')



# slider to move reference mirror
S = Slider(ax=ax[1,0], label='Time', valmin=0.8*1e-5, valmax=1.1*1e-5, valinit=0.95*1e-5)


mirror=ax[0,0].axvline(x=S.val, ymin=0.25, ymax=0.75, color='black', linestyle='--')
target = ax[0,0].plot([-0.2e-5,0.2e-5],[0.95e-5,0.95e-5], color='green', linewidth=3)


primaryBeam = ax[0,0].plot(x[:np.argwhere(x >= S.val)[0][0]], y[:np.argwhere(x >= S.val)[0][0]],color='blue')




reflectedBeam = ax[0,0].plot(2*S.val-x[np.argwhere(x >= S.val)[0][0]:np.argwhere(x >= 2*S.val)[0][0]], y[np.argwhere(x >= S.val)[0][0]:np.argwhere(x >= 2*S.val)[0][0]], color='red')

verticallyreflectedBeam = ax[0,0].plot(-y[np.argwhere(x >= 0)[0][0]:np.argwhere(x < 0.95*1e-5)[-1][-1]],x[np.argwhere(x >= 0)[0][0]:np.argwhere(x < 0.95*1e-5)[-1][-1]], color='blue')
verticallyreflectedBeamDown = ax[0,0].plot(-y[np.argwhere(x >= 0.95e-5)[0][0]:np.argwhere(x < 1.9*1e-5)[-1][-1]],1.9*1e-5-x[np.argwhere(x >= 0.95*1e-5)[0][0]:np.argwhere(x < 1.9*1e-5)[-1][-1]], color='red')
if Destructive:
    Combined_beam = ax[0,0].plot(np.zeros_like(finaly), -x[Nfinal:2*Nfinal], color='cyan')
else:
    Combined_beam = ax[0,0].plot(2*finaly, -x[Nfinal:2*Nfinal], color='purple')



ax[0,0].plot([-0.2*10**-5, 0.2*10**-5], [-0.2*10**-5, 0.2*10**-5],linewidth=10, color='black')






a= ax[0,1].plot(Delta,Is)
highlight = ax[0,1].axvline(x=0, color='red')

ax[0,1].set_xlabel('Shift relative to target (m)')
ax[0,1].set_ylabel('Intensity (a.u.)')

# make descriptive text in plot
ax[0,0].text(0.2*10**-5, 0.2*10**-5, 'Beamsplitter', fontsize=12, ha='left')
ax[0,0].text(0.85*10**-5, -0.1*10**-5, 'Mirror', fontsize=12, ha='center')
ax[0,0].text(0.2*10**-5, 0.95*10**-5, 'Target', fontsize=12, color='green', ha='left')
ax[0,0].text(0.1*10**-5, -0.5*10**-5, 'Interfering light', color='purple', fontsize=12, ha='left')

# Animating function, updates the plot when the slider is moved
def update(val):
    mirror.set_xdata([S.val, S.val])
    primaryBeam[0].set_xdata(x[:np.argwhere(x >= S.val)[0][0]])
    primaryBeam[0].set_ydata(y[:np.argwhere(x >= S.val)[0][0]])

    reflectedxdata = 2*S.val-x[np.argwhere(x >= S.val)[0][0]:]
    reflectedydata = y[np.argwhere(x >= S.val)[0][0]:].copy()
    index = np.argwhere(reflectedxdata > reflectedydata)

    reflectedxdata = reflectedxdata[index]
    reflectedydata = reflectedydata[index]

    reflectedBeam[0].set_xdata(reflectedxdata)
    reflectedBeam[0].set_ydata(reflectedydata)

    
    shift = int(2*(0.95e-5-S.val)//dx)
    Signal = Combined_signal(finaly, finalpol,np.roll(finaly,shift), np.roll(finalpol,shift))
    Combined_beam[0].set_xdata(Signal)
    highlight.set_xdata([S.val-0.95e-5])

    plt.draw()
    


S.on_changed(update)


plt.show()