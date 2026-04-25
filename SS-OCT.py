import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# file name for saving the animation
simname = "two_target_simulation"

# Simulation parameters
mirror = 1e-6 # Mirror position (in meters)
DeltaX = np.array([ 25e-6,60e-6]) # Example path length differences to targets (in meters)
Reflectivity = np.array([0.5, 0.5]) # Example reflectivity values for targets (0 to 1)


xMin = -1e-6 # Minimum x-value for the plot (1 micrometers)
xMax = 60e-6 # Maximum x-value for the plot (10 micrometers)
NPoints = 10000 # Number of points in the plot

if np.any(xMax < DeltaX):
    raise ValueError("All path length differences must be less than the maximum allowed (DeltaXmax).")


LambdaMin = 800e-9 # Minimum wavelength (400 nm)
LambdaMax = 1300e-9 # Maximum wavelength (800 nm)


#create wavenumbers space
Nks = 1000
ksToSimulate = np.linspace(1/LambdaMax, 1/LambdaMin, Nks) # Wavenumbers corresponding to wavelengths less than LambdaMax
intensities = np.zeros(2*Nks+1)

#create x-values and waveforms
x = np.linspace(xMin, xMax*3, NPoints)
Waves = ksToSimulate[:,np.newaxis]@x[np.newaxis,:] 
Waves = np.sin(2*np.pi*Waves) # Convert to sine waves

#Calculate intensities for each wavenumber
amplitudes = np.ones_like(ksToSimulate) # Assuming equal amplitude for all wavenumbers
Phases = np.zeros_like(ksToSimulate) # Assuming zero initial phase for simplicity (and without loss of generality)
for i in range(len(DeltaX)):
    path_diff = 2*DeltaX[i] # Path length difference for the target
    phase_shift = ksToSimulate*path_diff # Phase shift due to path length difference
    
    remaining_Amp = np.sqrt(np.prod(1-Reflectivity[:i]**2))
    reflectedAmp = remaining_Amp * Reflectivity[i]
    amplitudes = amplitudes * np.sqrt(1+reflectedAmp**2+2*reflectedAmp*np.cos(phase_shift - Phases)) # Scale amplitude by reflectivity

    Phases += np.arctan(reflectedAmp*np.sin(phase_shift - Phases)/(1+reflectedAmp*np.cos(phase_shift - Phases))) # Update phase based on reflection

for i in range(len(ksToSimulate)):
    intensities[i+1] = amplitudes[i]**2 # Intensity is proportional to the square of the amplitude
    intensities[-i] = amplitudes[i]**2 # Symmetric intensity for negative wavenumbers


# 1. Setup figure
fig, ax = plt.subplots(1,2, figsize=(10,5))

line, = ax[0].plot(x, Waves[0], color='blue')
ampdot, = ax[1].plot(ksToSimulate[1], amplitudes[1],'go')
amptail, = ax[1].plot(ksToSimulate[:1], amplitudes[:1], color='red')

ax[1].set_xlabel('Wavenumber (1/m)')
ax[1].set_ylabel('Amplitude (a.u.)')
ax[1].set_xlim(ksToSimulate[0], ksToSimulate[-1])
ax[1].set_ylim(0, np.max(amplitudes)*1.1)
ax[1].set_title('Amplitude vs Wavenumber')

# 2. Function to update each frame
def update(frame):
    line.set_ydata(Waves[frame])
    ampdot.set_data([ksToSimulate[frame]], [amplitudes[frame]])
    amptail.set_data(ksToSimulate[:frame], amplitudes[:frame])
    return line,

# 3. Create animation
ani = FuncAnimation(fig, update, frames=range(Waves.shape[0]), interval=1)

# 4. Save as GIF using the [PillowWriter](https://matplotlib.org)
ani.save(simname + ".gif", writer=PillowWriter(fps=30))

plt.show()

#show intensity spectrum
Ints = amplitudes**2
Ints -= np.mean(Ints) # Remove DC component for better visualization
plt.plot(ksToSimulate, Ints)
plt.show()

# show magnitude of the Fourier transform of the intensity spectrum
M=20 # Number of points to show in the Fourier transform plot
compWave = np.fft.fft(Ints)
compWave = compWave[:len(compWave)//2] # Take only the positive frequencies
# calcxs = 4*np.pi*500e-9*np.array(range(len(compWave))) # Convert FFT indices to path length differences (in meters)
calcxs = np.pi*((1/LambdaMin - 1/LambdaMax)**(-1))*np.array(range(len(compWave))) # Convert FFT indices to path length differences (in meters)

plt.plot(calcxs[:M], np.abs(compWave[:M]),'H-')
plt.xlabel('Path Length Difference (m)')
plt.ylabel('Magnitude of Fourier Transform (a.u.)')
plt.xticks(calcxs[:M+1])
labels = [f'{x*1e6:.1f}*pi μm' for x in np.arange(0,calcxs[M+1],np.pi*((1/LambdaMin - 1/LambdaMax)**(-1)))]
plt.gca().set_xticklabels(labels, rotation=70)
for i in range(len(DeltaX)):
    plt.axvline(DeltaX[i], color='red', linestyle='--', label=f'Target {i+1} (Δ={DeltaX[i]*1e6:.1f} μm)')
plt.legend()
plt.grid()
plt.show()


