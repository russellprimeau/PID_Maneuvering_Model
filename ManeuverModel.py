import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Task A: Implement a function which returns thrust as a function of prop angular speed

k_p = 1.47e5  # Propeller thrust coefficient for positive values, N*s*s
k_n = 1.65e5  # Propeller thrust coefficient for negative values, N*s*s
omega_range = np.arange(-132, 133, 1)  # Shaft speed, RPM
thrust_range = np.zeros(len(omega_range))  # Thrust force in surge direction, N


def Thrust(k_p_Thrust, k_n_Thrust, omega_Thrust):
    # k_p_Thrust = positive thrust coefficient, N*s*s
    # k_n_Thrust = negative thrust coefficient, N*s*s
    # omega_Thrust = shaft speed, RPM
    # Returns force in the positive x_B direction (thrust), N
    if omega_Thrust >= 0:
        if omega_Thrust > 132:
            omega_Thrust = 132
        omega_Thrust = omega_Thrust/60
        T_0 = k_p_Thrust * omega_Thrust ** 2
    else:
        if omega_Thrust < -132:
            omega_Thrust = -132
        omega_Thrust = omega_Thrust / 60
        T_0 = k_n_Thrust * abs(omega_Thrust) * omega_Thrust
    return T_0


# Plot thrust as a function of shaft speed over the full range of the ship's shaft speed outputs

for i in range(len(omega_range)):
    thrust_range[i, ] = Thrust(k_p, k_n, omega_range[i, ])

figA = plt.figure()
plt.tight_layout()
plt.plot(omega_range, thrust_range/1000, label='Thrust per propeller')
plt.xlabel('omega, RPM')
plt.ylabel('Thrust, kN')
plt.legend()
plt.grid()
plt.savefig('ProjectTaskA_Thrust.png')

##################################################################################################################

# Task B: Develop a polynomial function which describes the rudder lift and thrust coefficients as functions of the
# rudder angle, based on the available data.

# Datapoints written to CSV .txt from the image in the problem statement using "Engauge Digitizer" tool.

# Read data in from CSV file and write to a dataframe.
curves = pd.read_csv('LDdata.csv', sep=',', header=0)
# print(curves.Lift.to_string)

# Use Numpy's least squares function to find polynomials
f_L = np.polyfit(curves.x, curves.Lift, 5)
f_D = np.polyfit(curves.x, curves.Drag, 5)

# Test that function returns correct output
# print("Lift =", Thrust(k_p, k_n, 100) * np.polyval(f_L, 0))
# print("Thrust-Drag =", Thrust(k_p, k_n, 100) * np.polyval(f_D, 0))

# Calculate a correction term for the lift coefficient function, so that there is no lift force at delta = 0
lift_correct = -np.polyval(f_L, 0)

# Plot results to check that polynomial order is sufficient for accurate results
delta_range = np.arange(-45, 46, 1)
lift_coef = np.polyval(f_L, delta_range)
drag_coef = np.polyval(f_D, delta_range)

figB = plt.figure()
plt.tight_layout()
plt.plot(curves.x, curves.Lift, label='Lift Data')
plt.plot(curves.x, curves.Drag, label='Drag Data')
plt.plot(delta_range, lift_coef, label='Calculated Lift')
plt.plot(delta_range, drag_coef, label='Calculated Drag')
plt.xlabel('Rudder angle, degrees')
plt.ylabel('Force, % Thrust')
plt.legend()
plt.grid()
plt.savefig('ProjectTaskB_Lift,Drag.png')

###########################################################################################################

# Task C: Implement a function which returns a vector of surge force, sway force, and yaw torque


def get_force_torque(omega_getF, delta_getF):
    # omega_getF = shaft speed, rpm
    # delta_getF = rudder angle, degrees
    # Returns a vector [surge force, sway force, yaw torque]'
    F = np.zeros((3, 1))
    left = np.array([-41.5, 7.1,  0])
    right = np.array([-41.5, -7.1,  0])
    F[0, 0] = 2 * Thrust(k_p, k_n, omega_getF) * np.polyval(f_D, delta_getF)  # Surge force, N
    F[1, 0] = 2 * Thrust(k_p, k_n, omega_getF) * (np.polyval(f_L, delta_getF) + 0.0009873051556510198)  # Sway force,
    # N
    net_vec = np.array([F[0, 0]/2, F[1, 0]/2, 0])
    product = np.cross(left, net_vec) + np.cross(right, net_vec)
    F[2, 0] = product[2]  # Yaw torque, N*m
    return F


F_test = get_force_torque(100, 0)

# Plot function output over the full range of inputs for rudder angle to verify functionality

test_force_torque = np.zeros((3, 91))
for i in range(len(delta_range)):
    test_force_torque[0:3, i:i+1] = get_force_torque(100, delta_range[i])

figC, axsC = plt.subplots(2, 1)
plt.tight_layout()
axsC[0].set_ylabel('Force, kN')
axsC[0].plot(delta_range, test_force_torque[0, :]/1000, label='F_x, Surge Force at 100 rpm')
axsC[0].plot(delta_range, test_force_torque[1, :]/1000, label='F_y, Sway Force at 100 rpm')
axsC[0].grid(True)
axsC[0].legend()
axsC[1].set_ylabel('Torque, kNm')
axsC[1].set_xlabel('Rudder angle, degrees')
axsC[1].plot(delta_range, test_force_torque[2, :]/1000, label='Tau_z, Yaw Torque at 100 rpm')
axsC[1].grid(True)
axsC[1].legend()
figC.tight_layout()
plt.savefig('ProjectTaskC_Force,Torque.png')

###########################################################################################################

# Task D: Implement a system of first-order differential equations in matrix notation which defines the
# derivative of the state vector.

# Declare all variables given in the problem statement

M = np.array(([1.1e7, 0, 0], [0, 1.1e7, 8.4e6], [0, 8.4e6, 5.8e9]))  # mass matrix of the ship, kg
M_inv = np.linalg.inv(M)  # inversion of mass matrix of ship for solving for acceleration, kg
D = np.array(([3.0e5, 0, 0], [0, 5.5e5, 6.4e5], [0, 6.4e5, 1.2e8]))  # damping matrix of ship, kg/s
env = np.array(([200000], [0], [0]))  # environmental force in the world frame
A = np.zeros((6, 6))  # Create combined rotation matrix and (M^-1)(-D) coefficients for system of difEQs
A[3:, 3:] = M_inv@-D  # Divide -D by M and write to the corresponding quadrant of A for solving system of difEQs


def get_rotation(phi_r):
    # phi = heading, rad
    # Returns 3x3 rotation matrix for converting from body reference frame to world reference frame
    return np.array(([np.cos(phi_r), -np.sin(phi_r), 0], [np.sin(phi_r), np.cos(phi_r), 0], [0, 0, 1]))


def get_F_w(phi_w, env_F):
    # phi = heading, rad
    # env_F = 3x1 vector of environmental force in the world frame
    # Returns 3x1 vector of environmental forces and torques in the body frame [F_x, F_y, tau_Z]'
    return np.linalg.inv(get_rotation(phi_w))@env_F


def get_dot_state(A_s, X_s, omega_s, delta_s, M_inv_s, env_s):
    # A_s = Combined rotation matrix and (M^-1)(-D) coefficients for system of difEQs
    # X_s = 6x1 state vector of ship position and heading in world frame, ship velocity and rate of change in heading
    # n body frame [x, y, phi, surge, sway, rate of change in heading]', [m, m, rad, m/s, m/s, rad/s]'
    # omega_s = shaft speed, rpm
    # delta_s = rudder angle, degrees
    # env_s = environmental force
    # M_inv = inverse mass matrix, 1/kg
    # Returns the derivative of the state vector
    X_s[2, 0] = np.arctan2(np.sin(X_s[2, 0]), np.cos(X_s[2, 0]))  # Map the heading to the interval (-pi/2, pi/2)
    A_s[0:3, 3:] = get_rotation(X_s[2, 0])
    force = get_force_torque(omega_s, delta_s) + get_F_w(X_s[2, 0], env_s)
    return A_s@X_s + np.concatenate((np.zeros((3, 1)), M_inv_s@force), axis=0)


###########################################################################################################

# Task E: Simulate and plot the ship's trajectory over 1000 s.

# Initial values from problem statement:
omega_E = 100  # RPM
dt = .1  # Time step, s
time = np.arange(0, 1000, dt)  # Time range, s
X_sol = np.zeros((6, len(time)))  # Declare vector of state variables and initialize with values
                                    # from the problem statement
X_dot = np.zeros((6, len(time)))  # Declare vector of derivatives of state variables, to be computed
F_vec = np.zeros((3, len(time)))  # Declare vector of ship forces for debugging

# Define a function which calculates the rudder angle at each timestep.


def get_delta(t):
    # t = time, s
    # Returns the rudder angle at time t in degrees
    return 30*np.sin(0.06*t)


# Compute the values of delta at each timestep, for plotting
delta_E = np.zeros((len(time)))
for i in range(len(time)):
    delta_E[i] = get_delta(time[i])

# Iterate through time range using 4th order Runge-Kutta method to solve system of 1st order non-linear ordinary
# differential equations by numerical integration:
j = 1
while j <= len(time)-1:
    F_vec[0:3, j:(j+1)] = get_force_torque(omega_E, delta_E[j])  # Save out force for debugging
    k1 = get_dot_state(A, X_sol[0:6, (j-1):j], omega_E, delta_E[j-1], M_inv, env)
    k2 = get_dot_state(A, X_sol[0:6, (j-1):j] + k1*dt/2, omega_E, get_delta(time[j-1]+dt/2), M_inv, env)
    k3 = get_dot_state(A, X_sol[0:6, (j-1):j] + k2*dt/2, omega_E, get_delta(time[j-1]+dt/2), M_inv, env)
    k4 = get_dot_state(A, X_sol[0:6, (j-1):j] + k3*dt, omega_E, get_delta(time[j]), M_inv, env)
    X_dot[0:6, j:(j+1)] = (1/6)*(k1 + 2*k2 + 2*k3 + k4)  # Calculate dX/dt
    X_sol[0:6, j:(j+1)] = X_sol[0:6, (j-1):j] + dt*X_dot[0:6, j:(j+1)]  # Integrate to find next value of X
    j += 1  # Increment counter to move to next timestep

# Plot the solution variables vs. time and space to verify expected results
figE1, axsE1 = plt.subplots(2, 1)
plt.tight_layout(pad=2)
axsE1[0].plot(time, X_dot[0, :], label='X, world frame', marker=".")
axsE1[0].plot(time, X_dot[1, :], label='Y, world frame', marker=".")
axsE1[0].plot(time, X_sol[3, :], label='X, body frame (surge)')
axsE1[0].plot(time, X_sol[4, :], label='Y, body frame (sway)')
axsE1[0].set_xlabel('time, s')
axsE1[0].set_ylabel('velocity, m/s')
axsE1[0].legend()
axsE1[0].grid(True)

axsE1[1].plot(time, X_dot[3, :], label='X, body frame (surge)')
axsE1[1].plot(time, X_dot[4, :], label='Y, body frame (sway)')
axsE1[1].set_xlabel('time, s')
axsE1[1].set_ylabel('acceleration, m/s/s')
axsE1[1].legend()
axsE1[1].grid(True)
figE1.set_figwidth(8)
figE1.set_figheight(8)
plt.savefig('ProjectTaskE_derivatives.png')

figE2, axsE2 = plt.subplots(2, 1)
plt.tight_layout(pad=2)
axsE2[0].plot(time, X_sol[2, :]*180/np.pi, label='Phi, world frame heading')
axsE2[0].plot(time, delta_E, label='Rudder angle')
axsE2[0].set_xlabel('time, s')
axsE2[0].set_ylabel('Degrees')
axsE2[0].legend()
axsE2[0].grid(True)

axsE2[1].plot(time, X_dot[2, :], label='dPhi/dt, world frame', marker=".")
axsE2[1].plot(time, X_sol[5, :], label='dPhi/dt, body frame')
axsE2[1].set_xlabel('time, s')
axsE2[1].set_ylabel('rad/s')
axsE2[1].legend()
axsE2[1].grid(True)

figE2.set_figwidth(8)
figE2.set_figheight(8)
plt.savefig('ProjectTaskE_directions.png')

# Plot the ship's trajectory over the simulation period
figE3, axsE3 = plt.subplots(2, 1)
plt.tight_layout(pad=2)
axsE3[0].plot(X_sol[0, :], X_sol[1, :], label='Ship position')
axsE3[0].set_xlabel('x position, m')
axsE3[0].set_ylabel('y position, m')
axsE3[0].legend()
axsE3[0].axis('equal')
axsE3[0].grid(True)

axsE3[1].plot(X_sol[0, :], np.degrees(X_sol[2, :]), label='Heading')
axsE3[1].plot(X_sol[0, :], delta_E, label='Rudder angle')
axsE3[1].set_xlabel('x position, m')
axsE3[1].set_ylabel('degrees')
axsE3[1].legend()
axsE3[1].grid(True)
figE3.set_figwidth(8)
figE3.set_figheight(8)
plt.savefig('ProjectTaskE_position.png')

figE4, axsE4 = plt.subplots(2, 1)
plt.tight_layout(pad=2)
axsE4[0].plot(time, F_vec[0, :]/1000, label='X, boat frame')
axsE4[0].plot(time, F_vec[1, :]/1000, label='Y, boat frame')
axsE4[0].set_xlabel('time, s')
axsE4[0].set_ylabel('force, kN')
axsE4[0].legend()
axsE4[0].grid(True)

axsE4[1].plot(time, F_vec[2, :]/1000, label='Z, boat frame')
axsE4[1].set_xlabel('time, s')
axsE4[1].set_ylabel('torque, kNm')
axsE4[1].legend()
axsE4[1].grid(True)
figE4.set_figwidth(9)
figE4.set_figheight(8)
plt.savefig('ProjectTaskE_forces.png')

###########################################################################################################

# Task F: Implement a PID controller for the rudder angle which will keep the ship on course along a constant
# heading phi_0 = 30 degrees

# Initial values from problem statement:
omega_F = 100  # RPM
dt_F = .1  # Time step, s
time_F = np.arange(0, 500, dt_F)  # Time range, s
X_solF = np.zeros((6, len(time_F)))  # Declare vector of state variables and initialize with values
                                        # from the problem statement
X_dotF = np.zeros((6, len(time_F)))  # Declare vector of derivatives of state variables, to be computed
F_vecF = np.zeros((3, len(time_F)))  # Declare vector of ship forces for debugging
delta_F = np.zeros((len(time_F)))  # Declare vector for saving out the rudder angles, for tuning parameters

# Declare and initialize the variables needed for the PID controller
K_P = 2  # Proportional gain
K_I = 0.01  # Integral gain
K_D = 20  # Derivative gain
phi_0 = np.radians(30)  # Set point for heading, 30 degrees = pi/6 rad
Err = np.zeros((len(time_F)))
Err_int = np.zeros((len(time_F)))
Err_der = np.zeros((len(time_F)))
Err[0] = np.degrees(phi_0 - X_solF[2, 0])  # Error signal initial value

# Iterate through time range using 4th order Runge-Kutta method to solve system of 1st order non-linear ordinary
# differential equations by numerical integration, with the control input (delta) updated every timestep to
j = 1
while j <= len(time_F) - 1:
    # Calculate error signal (proportional)
    Err[j] = np.degrees(phi_0 - X_solF[2, j-1])

    # Calculate integral (cumulative) error
    Err_int[j] = (Err[j] + Err_int[j-1])

    # Calculate derivative of error
    Err_der[j] = (Err[j] - Err[j-1]) / dt_F

    # Calculate the control input
    PID = -K_P * Err[j] - K_I * Err_int[j] * dt_F - K_D * Err_der[j]

    # Limit rudder angle to min/max value
    if PID >= 0:
        delta_F[j] = min(PID, 45)
    else:
        delta_F[j] = max(PID, -45)

    # Compute the value for the current timestep based on the control inputs
    F_vecF[0:3, j:(j + 1)] = get_force_torque(omega_F, delta_F[j])  # Save out force for debugging
    k1 = get_dot_state(A, X_solF[0:6, (j - 1):j], omega_F, delta_F[j], M_inv, env)
    k2 = get_dot_state(A, X_solF[0:6, (j - 1):j] + k1 * dt_F / 2, omega_F, delta_F[j], M_inv, env)
    k3 = get_dot_state(A, X_solF[0:6, (j - 1):j] + k2 * dt_F / 2, omega_F, delta_F[j], M_inv, env)
    k4 = get_dot_state(A, X_solF[0:6, (j - 1):j] + k3 * dt_F, omega_F, delta_F[j], M_inv, env)
    X_dotF[0:6, j:(j + 1)] = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # Calculate dX/dt
    X_solF[0:6, j:(j + 1)] = X_solF[0:6, (j - 1):j] + dt_F * X_dotF[0:6, j:(j + 1)]  # Integrate to find next value of X
    j += 1  # Increment counter to move to next timestep

# Plot the solution variables vs. time to verify expected results
figF1, axsF1 = plt.subplots(2, 1)
plt.tight_layout(pad=2)
axsF1[0].plot(time_F, X_dotF[0, :], label='X, world frame', marker=".")
axsF1[0].plot(time_F, X_dotF[1, :], label='Y, world frame', marker=".")
axsF1[0].plot(time_F, X_solF[3, :], label='X, body frame (surge)')
axsF1[0].plot(time_F, X_solF[4, :], label='Y, body frame (sway)')
axsF1[0].set_xlabel('time, s')
axsF1[0].set_ylabel('velocity, m/s')
axsF1[0].legend()
axsF1[0].grid(True)

axsF1[1].plot(time_F, X_dotF[3, :], label='X, body frame (surge)')
axsF1[1].plot(time_F, X_dotF[4, :], label='Y, body frame (sway)')
axsF1[1].set_xlabel('time, s')
axsF1[1].set_ylabel('acceleration, m/s/s')
axsF1[1].legend()
axsF1[1].grid(True)
figF1.set_figwidth(8)
figF1.set_figheight(8)
plt.savefig('ProjectTaskF_derivatives.png')

figF2, axsF2 = plt.subplots(2, 1)
plt.tight_layout(pad=2)
axsF2[0].plot(time_F, X_solF[2, :]*180/np.pi, label='Phi, world frame heading')
axsF2[0].plot(time_F, delta_F, label='Rudder angle')
axsF2[0].set_xlabel('time, s')
axsF2[0].set_ylabel('Degrees')
axsF2[0].legend()
axsF2[0].grid(True)

axsF2[1].plot(time_F, X_dotF[2, :], label='dPhi/dt, world frame', marker=".")
axsF2[1].plot(time_F, X_solF[5, :], label='dPhi/dt, body frame')
axsF2[1].set_xlabel('time, s')
axsF2[1].set_ylabel('rad/s')
axsF2[1].legend()
axsF2[1].grid(True)

figF2.set_figwidth(9)
figF2.set_figheight(8)
plt.savefig('ProjectTaskF_directions.png')

# Plot the ship's trajectory in space the simulation period
figF3, axsF3 = plt.subplots(2, 1)
plt.tight_layout(pad=2)
axsF3[0].plot(X_solF[0, :], X_solF[1, :], label='Ship position')
axsF3[0].set_xlabel('x position, m')
axsF3[0].set_ylabel('y position, m')
axsF3[0].legend()
axsF3[0].axis('equal')
axsF3[0].grid(True)

axsF3[1].plot(X_solF[0, :], np.degrees(X_solF[2, :]), label='Heading')
axsF3[1].plot(X_solF[0, :], delta_F, label='Rudder angle')
axsF3[1].set_xlabel('x position, m')
axsF3[1].set_ylabel('degrees')
axsF3[1].legend()
axsF3[1].grid(True)
figF3.set_figwidth(8)
figF3.set_figheight(8)
plt.savefig('ProjectTaskF_position.png')

figF4, axsF4 = plt.subplots(2, 1)
plt.tight_layout(pad=2)
axsF4[0].plot(time_F, Err, label='Error (Proportional)')
axsF4[0].plot(time_F, Err_int, label='Error (Integral)')
axsF4[0].plot(time_F, Err_der, label='Error (Derivative)')
axsF4[0].set_xlabel('time, s')
axsF4[0].set_ylabel('error signal, degrees')
axsF4[0].legend(loc='lower right')
axsF4[0].grid(True)

axsF4[1].plot(time_F, K_P * Err, label='Gain * Error (Proportional)')
axsF4[1].plot(time_F, K_I * Err_int, label='Gain * Error (Integral)')
axsF4[1].plot(time_F, K_D * Err_der, label='Gain * Error (Derivative)')
axsF4[1].plot(time_F, delta_F, label='Rudder Command')
axsF4[1].set_xlabel('time, s')
axsF4[1].set_ylabel('error signal, degrees')
axsF4[1].set_ylim(-90, 90)
axsF4[1].legend(loc='lower right')
axsF4[1].grid(True)
figF4.set_figwidth(8)
figF4.set_figheight(8)
plt.savefig('ProjectTaskF_PID.png')
