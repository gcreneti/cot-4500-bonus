import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

def make_diagonally_dominant(matrix, b_vector):
  n = len(matrix)
  
  for i in range(n):
    pivot: float = matrix[i][i]
    sum_of_other_elements = sum(abs(matrix[i][i+1:]))
    
    if abs(pivot) > abs(sum_of_other_elements):
      continue
    max_value_of_row = 0
    max_index_in_row = 0
    
    for j in range(n):
      current_value_in_row: float = abs(matrix[i][j])
      
      if current_value_in_row > max_value_of_row:
        max_value_of_row = current_value_in_row
        max_index_in_row = j
        
    matrix[[i, max_index_in_row]] = matrix[[max_index_in_row, i]]
    b_vector[[i, max_index_in_row]] = b_vector[[max_index_in_row, i]]
    
  return matrix, b_vector

def jacobi(matrix, b_vector,tol):
  a = matrix
  b = b_vector
  x1 = 0
  x2 = 0
  x3 = 0
  i = 2
  
  while(i<=50):
    new_x1 = (b[0] - a[0,1]*x2 - a[0,2]*x3)/3
    new_x2 = (b[1] - a[1,0]*x1 - a[1,2]*x3)/4
    new_x3 = (b[2] - a[2,0]*x1 - a[2,1]*x2)/7

    x1 = new_x1
    x2 = new_x2
    x3 = new_x3
    
    x_vector = np.array([round(x1,6),round(x2,6),round(x3,6)])
    x_exact = np.array([round(0.2,6),round(0.8,6),round(-0.4,6)])
    diff = abs(x_exact - x_vector)
    
    if(round(np.linalg.norm(diff),6) < tol):
      return i
      
    i += 1

def gauss_seidel(matrix, b_vector,tol):
  a = matrix
  b = b_vector
  x1 = 0
  x2 = 0
  x3 = 0
  i = 1
  
  while(i<=50):
    x1 = (b[0] - a[0,1]*x2 - a[0,2]*x3)/3
    x2 = (b[1] - a[1,0]*x1 - a[1,2]*x3)/4
    x3 = (b[2] - a[2,0]*x1 - a[2,1]*x2)/7
    x_vector = np.array([round(x1,6),round(x2,6),round(x3,6)])
    x_exact = np.array([0.200000,0.800000,-0.400000])
    diff = abs(x_exact - x_vector)
    
    if(np.linalg.norm(diff) < tol):
      return i
      
    i += 1

def custom_derivative(value):
  return (3 * value* value) - (2 * value) 
  
def newton_raphson(initial_approximation: float, tol: float, sequence: str):
  iteration_counter = 1
  x = initial_approximation
  f = eval(sequence)
  f_prime = custom_derivative(initial_approximation)
  approximation: float = f / f_prime
  while(abs(approximation) >= tol):
    x = initial_approximation
    f = eval(sequence)
    f_prime = custom_derivative(initial_approximation)
    approximation = f / f_prime
    initial_approximation -= approximation
    
    if(abs(approximation) < tol):
      return iteration_counter
      
    iteration_counter += 1

def apply_div_dif(matrix: np.array):
  size = len(matrix)
  for i in range(2, size):
    for j in range(2, i+2):
      if j >= len(matrix[i]) or matrix[i][j] != 0:
        continue
        
      left: float = matrix[i][j-1]
      diagonal_left: float = matrix[i-1][j-1]
      numerator: float = left-diagonal_left
      
      if(j<4):
        denominator = matrix[i][0]-matrix[i-2][0]
        
      elif(j>=4):
        denominator = matrix[i][0]-matrix[i-3][0]
        
      operation = numerator / denominator
      matrix[i][j] = operation
      
  return matrix
  
def hermite_interpolation():
  x_points = [0, 1, 2]
  y_points = [1, 2, 4]
  slopes = [1.06, 1.23, 1.55]
  num_of_points = len(x_points)
  matrix = np.zeros((6, 6))
  matrix[0][0] = 0
  matrix[1][0] = 0
  matrix[2][0] = 1
  matrix[3][0] = 1
  matrix[4][0] = 2
  matrix[5][0] = 2
  matrix[0][1] = 1
  matrix[1][1] = 1
  matrix[2][1] = 2
  matrix[3][1] = 2
  matrix[4][1] = 4
  matrix[5][1] = 4
  matrix[1][2] = 1.06
  matrix[3][2] = 1.23
  matrix[5][2] = 1.55
      
  filled_matrix = apply_div_dif(matrix)
  print(filled_matrix)

def euler_function(t: float, w: float):
  return w - (t**3)

def do_work(t, w, h):
  basic_function_call = euler_function(t, w)
  incremented_t = t + h
  incremented_w = w + (h * basic_function_call)
  incremented_function_call = euler_function(incremented_t, incremented_w)
  return basic_function_call + incremented_function_call

def modified_eulers():
  original_w = .5
  start_of_t, end_of_t = (0, 3)
  num_of_iterations = 100
  h = (end_of_t - start_of_t) / num_of_iterations
  
  for cur_iteration in range(0, num_of_iterations):
    t = start_of_t
    w = original_w
    h = h
    inner_math = do_work(t, w, h)
    next_w = w + ( (h / 2) * inner_math )
    start_of_t = t + h
    original_w = next_w
    
  return next_w

    

if __name__ == "__main__":
  
  matrix = np.array([[3,1,1],
                   [1,4,1],
                   [2,3,7]])
  
  b_vector = np.array([1,3,0])
  
  initial_approximation: float = 0.5
  
  sequence: str = "(x**3) - (x**2) + 2"
  
  tol: float = 0.000001
  d_matrix, new_b = make_diagonally_dominant(matrix, b_vector)
  
  gs_converge = gauss_seidel(d_matrix, new_b, tol)
  
  jac_converge = jacobi(d_matrix, new_b,tol)
  
  print(gs_converge, "\n")
  
  print(jac_converge, "\n")
  
  nr = newton_raphson(initial_approximation, tol, sequence)
  
  print(nr, "\n")
  
  hermite_interpolation()
  
  mod_eulers = modified_eulers()
  
  print(mod_eulers, "\n")

