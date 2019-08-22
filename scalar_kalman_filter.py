# Example of Kalman filtering for vehicle speed
# Joni Leppänen, 22.08.2019
#
# See the algorithm proof at:
# http://www.swarthmore.edu/NatSci/echeeve1/Ref/Kalman/ScalarKalman.html
#
# This script uses the same notation as in the website above except the
# Kalman gain is noted as 'G' instead of k, and the time index is noted
# as k instead of j. Also, do not confuse the sample period T in this code to
# the delay operator also noted as T in the block diagrams on the website.
import numpy as np
from matplotlib import pyplot as plt
import random

# Set seed for the random generator so that comparing the algorithm output when
# tweaking the parameters is easier.
random.seed(1)

# Simulation parameters
N = 300                         # Number of samples
T = 0.2                         # Sample period [seconds]
u = 4000                        # Gas (system input) [Newtons]

# Real vehicle parameters (the absolutely correct values for simulation)
m_real = 1500                   # Mass [kg]
theta_real = 200                # Air drag
w_real = 0.4                    # Process noise standard deviation
r_real = 2                      # Measurement noise standard deviation

# Constants (for state space model)
a_real = 1 - T*theta_real/m_real   # System constant
b_real = T/m_real                  # Input constant

# Kalman filter parameters (must be measured or estimated somehow)
# Vehicle model constants.
m = 1500                        # Mass [kg]
theta = 200                     # Air drag
a = 1 - T*theta/m               # System constant
b = T/m                         # Input constant

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
v = np.zeros((N, 1))  # Real speed with process noise
z = np.zeros((N, 1))  # Speed measurement with measurement noise

for k in range(0, N - 1):
    v[k+1] = a_real*v[k] + b_real*u + w_real*(2*random.random()-1)
    z[k+1] = v[k+1] + r_real*(2*random.random()-1) + bias

# Initialize the simulation variables. Store the estimates to vector for
# plotting.
out_x_hat = np.zeros((N, 1))
out_G = np.zeros((N, 1))
out_p = np.zeros((N, 1))

# The simulation loop where the Kalman filter is used.
for k in range(0, N):

    # OPTIONAL
    # If speed measurements stop, the measurement variance is set to very high
    # value so that the algorithm would give more weight to the model. So
    # for a while the speed is read only from the model hoping it won't drift
    # too far from the actual speed.
    if z[k] == 0:
        R = 200000
    else:
        R = r**2

    # ----------------------- THE KALMAN FILTER BEGINS -------------------------

    # Predict the next state using the system model (state space model).
    x_hat = a*x_hat + b*u       # a-priori estimate
    p = (a**2)*p + Q            # a-priori variance

    # Calculate the optimal gain (Kalman gain).
    G = (h*p)/((h**2)*p + R)

    # Correct the prediction using measurement and Kalman gain.
    x_hat = x_hat + G*(z[k]-h*x_hat)  # a-posteriori estimate
    p = p*(1-h*G)                     # a-posteriori variance

    # ----------------------- THE KALMAN FILTER ENDS ---------------------------

    # Collect data for plotting.
    out_x_hat[k] = x_hat
    out_G[k] = G
    out_p[k] = p


# Create time vector for plotting.
t = T * np.arange(N)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t, v, t, z, '.', t, out_x_hat)
plt.title('Car speed estimation using Kalman filter')
plt.legend(('Real speed', 'Measured speed', 'Estimated speed'))
plt.xlabel('Time [seconds]')
plt.ylabel('Speed [m/s]')
plt.grid()
plt.show()
