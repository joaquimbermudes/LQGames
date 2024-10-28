# Importing necessary libraries
import numpy as np
from scipy.linalg import expm
import copy

import warnings
import numpy as np
from numpy.linalg import inv, LinAlgError, norm, cond, svd

from scipy.linalg import solve, solve_triangular, matrix_balance
from scipy.linalg import get_lapack_funcs
from scipy.linalg import schur
from scipy.linalg import lu
from scipy.linalg import qr
from scipy.linalg import ordqz
from scipy.linalg import kron, block_diag

# Auxiliary function that multiplies corresponding elements in two arrays A and B of the same length
# Each element is an np.array, and this function performs matrix multiplication for each pair
def Produto(A, B, n):
  C = [None] * n
  for i in np.arange(n):
    C[i] = A[i] @ B[i]  # Matrix multiplication for each element in the arrays

  return C

# Alternative auxiliary function that performs element-wise multiplication for corresponding items in two arrays
def Produto2(A, B, n):
  C = []
  for i in np.arange(n):
    C.append(A[i] * B[i])  # Element-wise multiplication

  return C

# Auxiliary function that calculates the distance between components of two arrays A and B of the same length
# This uses a norm-based approach to sum the absolute differences
def Distancia(A, B, n):
  C = []  # Initialize C as an empty list to store individual distances
  for i in np.arange(n):
    C.append(np.sum(np.abs(A[i] - B[i])))  # Append each distance to the list

  return sum(C)  # Sum all distances in the list to get the total distance

# Function to solve a continuous algebraic Riccati equation in a game setup
def solve_continuous_are_game(a, Others_Players, b, q, r, e=None, s=None, balanced=True):

    # Set up problem dimensions and variable types
    m = np.shape(a)[0]
    n = np.shape(b)[1]

    # Determine whether to use complex or float data types
    r_or_c = complex if r.dtype == complex else float
    gen_or_not = False
    gen_are = True if e is not None else False

    # Set up the Hamiltonian matrix H based on inputs
    H = np.empty((2*m+n, 2*m+n))
    H[:m, :m] = a + Others_Players
    H[:m, m:2*m] = 0.
    H[:m, 2*m:] = b
    H[m:2*m, :m] = -q
    H[m:2*m, m:2*m] = -a.conj().T
    H[m:2*m, 2*m:] = 0. if s is None else -s
    H[2*m:, :m] = 0. if s is None else s.conj().T
    H[2*m:, m:2*m] = b.conj().T
    H[2*m:, 2*m:] = r

    # Configure matrix J based on the generalized ARE setting
    if gen_are and e is not None:
        J = block_diag(e, e.conj().T, np.zeros_like(r))
    else:
        J = block_diag(np.eye(2*m), np.zeros_like(r))

    # Balance the Hamiltonian pencil (H, J) if specified
    if balanced:
        # Calculate balancing factors
        M = np.abs(H) + np.abs(J)
        np.fill_diagonal(M, 0.)
        _, (sca, _) = matrix_balance(M, separate=1, permute=0)
        if not np.allclose(sca, np.ones_like(sca)):
            sca = np.log2(sca)
            s = np.round((sca[m:2*m] - sca[:m])/2)
            sca = 2 ** np.r_[s, -s, sca[2*m:]]
            elwisescale = sca[:, None] * np.reciprocal(sca)
            H *= elwisescale
            J *= elwisescale

    # Deflate the pencil to a smaller matrix
    q, r = qr(H[:, -n:])
    H = q[:, n:].conj().T.dot(H[:, :2*m])
    J = q[:2*m, n:].conj().T.dot(J[:2*m, :2*m])

    out_str = 'real' if r_or_c == float else 'complex'

    _, _, _, _, _, u = ordqz(H, J, sort='lhp', overwrite_a=True,
                             overwrite_b=True, check_finite=False,
                             output=out_str)

    # Obtain stable subspace basis
    if e is not None:
        u, _ = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
    u00 = u[:m, :m]
    u10 = u[m:, :m]

    # Solve triangular matrix system
    up, ul, uu = lu(u00)
    if 1/cond(uu) < np.spacing(1.):
        raise LinAlgError('Failed to find a finite solution.')

    x = solve_triangular(ul.conj().T,
                         solve_triangular(uu.conj().T,
                                          u10.conj().T,
                                          lower=True),
                         unit_diagonal=True,
                         ).conj().T.dot(up.conj().T)
    if balanced:
        x *= sca[:m, None] * sca[:m]

    # Check symmetry condition to ensure a valid solution
    u_sym = u00.conj().T.dot(u10)
    n_u_sym = norm(u_sym, 1)
    u_sym = u_sym - u_sym.conj().T
    sym_threshold = np.max([np.spacing(1000.), 0.1*n_u_sym])

    if norm(u_sym, 1) > sym_threshold:
        raise LinAlgError('The associated Hamiltonian pencil has eigenvalues '
                          'too close to the imaginary axis')

    return (x + x.conj().T)/2  # Return the symmetric solution

# Function to solve the open-loop CARE (Continuous Algebraic Riccati Equation)
def Open_Loop_CARE(A, B, Q, R, tolerance = 0.0001):
  """
  Parameters:
  A = autonomous system movement matrix
  B = array of input matrices for all players
  Q = array of state weight matrices for all players
  R = array of input weight matrices for all players
  tolerance = solution error tolerance
  """
  A = np.matrix(A)
  n_players = len(B)
  D = 1000
  S = [None] * n_players
  X = [None] * n_players
  for i in np.arange(n_players):
    B[i] = np.matrix(B[i])
    R[i] = np.matrix(R[i])
    S[i] = - B[i] @ np.linalg.inv(R[i]) @ B[i].T
    X[i] = np.zeros(np.shape(A))
  Y = copy.deepcopy(X)
  k = 0
  while D > tolerance:
    k = k + 1
    for j in np.arange(n_players):
      Outros = sum(Produto(X, Produto(S, X, n_players), n_players)) + X[j]*S[j]*X[j]
      X[j] = solve_continuous_are_game(a = A, Others_Players = Outros, b = B[j], q = Q[j], r = R[j])
    D = Distancia(Y, X, n_players)
    Y = copy.deepcopy(X)
    if k > 1000:
      break

  return X  # Return controllers

# Function to simulate system evolution x over time given system parameters
def Simulation_x(A, B, R, P, x_0, M = 100, T_f = 20):
  N_players = len(B)

  S = [None] * N_players

  A = np.matrix(A)
  x_0 = np.matrix(x_0)

  for i in np.arange(N_players):
    B[i] = np.matrix(B[i])
    R[i] = np.matrix(R[i])
    S[i] = - B[i] @ np.linalg.inv(R[i]) @ B[i].T

  x = [None] * T_f * M
  delta_t = 1/M
  for t in np.arange(T_f * M):
    x[t] = expm( (A + sum(Produto(S, P, N_players))) * t * delta_t) @ x_0

  return np.reshape(x, (T_f * M, np.shape(A)[0]))

# Function to simulate control inputs u over time based on system states x
def Simulation_u(x, B, R, P, M = 100, T_f = 20):
  u = [None] * T_f * M
  B = np.matrix(B)
  R = np.matrix(R)
  P = np.matrix(P)

  RBP = np.linalg.inv(R) @ B.T @ P
  for t in np.arange(T_f * M):
    u[t] = - RBP @ x[t]

  return np.reshape(u, (T_f * M, np.shape(P)[0]))

# Function to solve a finite-time LQ game
def Solve_Finite_Time(A, Bs, Qs, Rs, Kfs, Tf, x_0, granularidade = 100):
  N_players = len(Qs)
  N_states = np.shape(A)[0]
  Ss = [None] * N_players

  for i in np.arange(N_players):
    Ss[i] = Bs[i] @ np.linalg.inv(Rs[i]) @ Bs[i].T

  M = np.zeros(((N_players + 1) * N_states, (N_players + 1) * N_states))

  M[0:N_states, 0:N_states] = -A

  for i in np.arange(1, N_players + 1):
    M[ i * N_states : (i + 1) * N_states, i * N_states : (i + 1) * N_states] = np.transpose(A)
    M[ i * N_states : (i + 1) * N_states, 0 : N_states] = Qs[i - 1]
    M[0 : N_states, i * N_states : (i + 1) * N_states] = Ss[i - 1]

  Norma = np.linalg.norm(M)

  W = expm(M * Tf)

  H = W[0:N_states, 0:N_states]

  for i in np.arange(1, N_players + 1):
    H = H + W[0 : N_states, i * N_states : (i+1) * N_states] @ Kfs[i - 1]

  IO = np.zeros((N_states, N_states * (N_players + 1)))
  IO[0:N_states, 0:N_states] = np.identity(N_states)

  IO2 = np.zeros((N_states * (N_players + 1), N_states))
  IO2[0:N_states, 0:N_states] = np.identity(N_states)

  for i in np.arange(1, N_players + 1):
    IO2[ N_states * i : N_states * (i + 1), 0:N_states] = Kfs[i - 1]

  x_t = [None] * Tf * granularidade
  y_0 =  IO2 @ np.linalg.inv(H) @ x_0

  for t in np.arange(Tf * granularidade):
    x_t[t] = (expm(M * (Tf - t/granularidade)) @ y_0)

  l = len(x_t[0])

  return np.reshape(x_t, (Tf * granularidade, l))

# Class to define an infinite-horizon LQ game
class LQGame_Infnity():
  def __init__(self, A, B, Q, R, x_0, r = 0, T_f = 20, tolerance = 0.001, M = 100):
    self.N_players = len(B)
    self.A = np.matrix(A)
    self.A_Sim = np.matrix(A + 0.5 * r * np.identity(np.shape(A)[0]))
    self.x_0 = np.matrix(x_0)
    self.B = [None] * self.N_players
    self.Q = [None] * self.N_players
    self.R = [None] * self.N_players
    for i in range(self.N_players):
      self.B[i] = np.matrix(B[i])
      self.Q[i] = np.matrix(Q[i])
      self.R[i] = np.matrix(R[i])
    self.T_f = T_f
    self.M = M
    self.P = Open_Loop_CARE(A = self.A, B = self.B, Q = self.Q, R = self.R, tolerance =  tolerance)

  # Function to simulate state evolution x
  def Simulation_x(self):
    self.Simulation_x = Simulation_x(A = self.A_Sim, B = self.B, R = self.R, P = self.P, x_0 = self.x_0, T_f = self.T_f, M = self.M)

  # Function to simulate control evolution u
  def Simulation_u(self):
    self.Simulation_u = [None] * self.N_players
    for i in np.arange(self.N_players):
      self.Simulation_u[i] = Simulation_u(x = self.Simulacao_x, B = self.B[i], R = self.R[i], P = self.P[i], T_f = self.T_f)


# Class to define a finite-horizon LQ game
class LQGame_Finity():
  def __init__(self, A, B, Q, R, Kf, x_0, T_f = 20, M = 100):
    self.N_players = len(B)
    self.A = np.matrix(A)
    self.x_0 = np.matrix(x_0)

    self.Kf = [None] * self.N_players
    self.B = [None] * self.N_players
    self.Q = [None] * self.N_players
    self.R = [None] * self.N_players

    for i in range(self.N_players):
      self.B[i] = np.matrix(B[i])
      self.Q[i] = np.matrix(Q[i])
      self.R[i] = np.matrix(R[i])
      self.Kf[i] = np.matrix(Kf[i])

    self.T_f = T_f
    self.M = M

  # Function to simulate the finite-time LQ game
  def Simulation(self):
    self.Simulation = Solve_Finite_Time(A = self.A, Qs = self.Q, Bs = self.B, Rs = self.R, Kfs = self.Kf, Tf = self.T_f, x_0 = self.x_0, granularidade = self.M)

  # Function to simulate both state and control evolution
  def Simulation_x_u(self):
    self.Simulation_x = [None] * self.T_f * self.M
    self.Simulation_u = [None] * self.N_players
    N_states = np.shape(self.A)[0]
    for t in np.arange(self.T_f * self.M):
      self.Simulation_x[t] = self.Simulation[t][0:N_states]

    self.Simulation_x = np.reshape(self.Simulaction_x, (self.T_f * self.M, N_states))

    for i in np.arange(1, self.N_players+1):
      RB = - np.linalg.inv(self.R[i-1]) @ self.B[i-1].T
      self.Simulation_u[i-1] = [None] * self.T_f * self.M
      for t in np.arange(self.T_f * self.M):
        self.Simulation_u[i-1][t]  = RB @ self.Simulation[t][i * N_states : (i + 1) * N_states]
      self.Simulation_u[i-1] = np.reshape(self.Simulation_u[i-1], (self.T_f * self.M, np.shape(RB)[0]))
