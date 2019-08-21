# Example of Kalman estimation for vehicle speed.
# Joni Leppänen
#
# See the algorithm proof at:
# http://www.swarthmore.edu/NatSci/echeeve1/Ref/Kalman/ScalarKalman.html
#
# This script uses the same notation as in the website above except the
# Kalman gain is noted with 'G' to avoid confusion with the time index k.
import numpy as np
from matplotlib import pyplot as plt
import random

# Seed for random generator
random.seed(1)

# Simulation parameters
n_samples = 350                 # Number of samples
T = 0.2                         # Sample period
u = 1500                        # Gas (system input) [Newtons]g

# Real vehicle parameters (the absolutely correct values for simulation)
m_real = 1500                   # Mass [kg]
theta_real = 200                # Air drag
w_real = 0.4                    # Process noise standard deviation
r_real = 2                      # Measurement noise standard deviation

# Constants (for state space model)
a_real = -T*theta_real/m_real + 1   # System constant
b_real = 1/m_real               # Measurement constant

# Kalman filter parameters (must be measured or estimated somehow)
# Vehicle model constants.
m = 1500                        # Mass [kg]
theta = 200                     # Air drag
a = (-T*theta/m + 1)            # System constant
b = 1/m                         # Measurement constant

# Noise parameters.
w = 0.4                         # Process noise standard deviation
r = 2                           # Measurement noise standard deviation
R = r**2                        # Measurement noise variance
Q = w**2                        # Process noise variance

bias = 0                        # Measurement bias
h = 1                           # Measurement gain

# Initial Kalman variance should be set to high so that the algorithm gives
# more weight to the measurements.
p = 1000                        # Initial Kalman variance

# Initial state (speed).
x_hat = 0

# Create the real vehicle speed vector and the measurement vector
v = np.zeros((n_samples + 1, 1))  # Real speed with process noise
z = np.zeros((n_samples + 1, 1))  # Speed measurement with measurement noise

for k in range(0, n_samples):
    v[k+1] = a_real*v[k] + b_real*u + w_real*(2*random.random()-1)
    z[k+1] = v[k+1] + r_real*(2*random.random()-1) + bias

# Initialize the simulation variables. Store the estimates to vector for
# plotting.
out_x_hat = np.zeros((n_samples+1, 1))
out_G = np.zeros((n_samples+1, 1))
out_p = np.zeros((n_samples+1, 1))

# Run the Kalman filter
for k in range(0, n_samples):
    pass

    # OPTIONAL
    # If speed measurements stop, the measurement variance is set to very high
    # value so that the algorithm would give more weight to the model. % So
    # for a while the speed is read only from the model hoping it won't drift
    # too far from the actual speed.
    if z[k] == 0:
        R = 200000
    else:
        R = r**2

    # THE KALMAN FILTER BEGINS

    # Predict the next state using the system model (state space model).
    x_hat = a*x_hat + b*u       # a-priori estimate
    p = (a**2)*p + Q            # a-priori variance

    # Calculate the optimal gain (Kalman gain).
    G = (h*p)/((h**2)*p + R)

    # Correct the prediction using measurement and Kalman gain.
    x_hat = x_hat + G*(z[k]-h*x_hat)  # a-posteriori estimate
    p = p*(1-h*G)                     # a-posteriori variance

    # KALMAN FILTER ENDS

    # Collect data for plotting.
    out_x_hat[k] = x_hat
    out_G[k] = G
    out_p[k] = p

t = T * np.arange(n_samples+1)

plt.figure(figsize=(12, 6))
plt.plot(t, v, t, z, '.', t, out_x_hat)
plt.legend(('Real speed', 'Measured speed', 'Estimated speed'))
plt.xlabel('Time [seconds]')
plt.ylabel('Speed [m/s]')
plt.grid()
plt.show()
