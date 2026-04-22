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


#Interference functions
# Intensity
def Intensity(x, polx,y, poly): 
    Inten = np.sum(((x+(y*np.cos(polx-poly+phaseshift)))**2)+((y*np.sin(polx-poly+phaseshift))**2))
    return Inten

# Leading component of the combined signal
def Combined_signal(x, polx,y, poly):
    xy_comb = x+y*np.cos(polx-poly+phaseshift)
    yx_comb = y*np.sin(polx-poly+phaseshift)
    indeces = np.abs(xy_comb)<np.abs(yx_comb)
    xy_comb[indeces] = yx_comb[indeces]
    return xy_comb



# Wave generator parameters
# wavelengths = (np.random.rand(20)*900+100)*10**(-9) # 100-1000 nm
wavelengths = 0.5*(np.random.rand(20)*500+800)*10**(-9) # 800-1300 nm
#wavelengths = 0.5*(np.random.rand(20)*200+400)*10**(-9) # 400-600 nm
norms = (np.random.rand(20)-0.5)*2
phases = np.random.rand(20)*2*np.pi
nmTom = 1e-9
polaritydiff_coeff = (2*np.pi)**2/((500*nmTom))
bounds = 1e-5
N_points = 100000

Nfinal = N_points//4



dx = 4*bounds/N_points
pstep = polaritydiff_coeff*dx 

norms /= np.sum(np.abs(norms))
Wavenumber = 1/wavelengths

# Wave function
wave = lambda x: np.sum(norms*np.sin(2*np.pi*Wavenumber*x + phases))


# Wave generation
x = np.linspace(-bounds, 3*bounds, N_points)
y = [wave(xi) for xi in x]
y = np.array(y)/(2*10**5)

# after beamsplitter, the amplitude of the wave is halved
y[x>0] /= 2

# Polarity modeled as a random walk, with step size determined by a diffusion coefficient and the resolution along the propagation direction. The diffusion coefficient is chosen to give a reasonable amount of polarity variation over the length of the simulation.
polarity = np.sign(np.random.rand(N_points) - 0.5)*pstep
polarity = np.cumsum(polarity)% (2*np.pi)


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





# Precalculate Intensity for the final beam
Steps = int((S.valmax - S.valmin)//dx)
Delta = np.linspace(-0.15e-5,0.15e-5,Steps)
Is = []
for i in range(Steps):
    shift = (i-Steps//2)*2
    baseline = y[Nfinal:3*Nfinal]
    infering = y[Nfinal+shift:3*Nfinal+shift] 
    basepol = polarity[Nfinal:3*Nfinal]
    inferpol = polarity[Nfinal+shift:3*Nfinal+shift]
    Is.append(Intensity(baseline, basepol, infering, inferpol))
Is = np.array(Is)
a= ax[0,1].plot(Delta,Is)
highlight = ax[0,1].axvline(x=0, color='red')

ax[0,1].set_xlabel('Shift relative to target (m)')
ax[0,1].set_ylabel('Intensity (a.u.)')

# make descriptive text in plot
ax[0,0].text(0.2*10**-5, 0.2*10**-5, 'Beamsplitter', fontsize=12, ha='left')
ax[0,0].text(0.1*10**-5, -0.1*10**-5, 'Mirror', fontsize=12, ha='center')
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
# savebutton = Button(ax[1,1], 'Save Waveform')
# def save_data(event):
#     np.savetxt('interference_data.csv', np.column_stack((Delta, Is)), delimiter=',', header='Shift (m), Intensity (a.u.)', comments='')
#     np.savetxt('waveform_data.csv', np.column_stack((x, y, polarity)), delimiter=',', header='Position (m), Amplitude (a.u.), Polarity (rad)', comments='')
# savebutton.on_clicked(save_data)

plt.show()