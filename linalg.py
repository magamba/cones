# -*- coding: utf-8 -*-
import numpy as np 

def distance(w1, b1, w2, b2, normalize = False):
  """ Distance between the apices of two polyhedral cones
      together with its sign.
      
      Let cone_i = cone(w_i, b_i) be the cone identified by
      w_i and b_i.
      
      If sign > 0 then cone_1 is (partially) contained in cone_2.
      if sign == 0 then the two apices coincide.
      If sign < 0 then cone_2 is (partially) contained in cone_1.
  """
  coef1 = -b1 / np.sum(w1)
  coef2 = -b2 / np.sum(w2)
  v1 = coef1 * np.ones_like(w1)
  v2 = coef2 * np.ones_like(w2)
  
  distance = l2norm(np.abs(v1 - v2))
  if normalize:
    distance = distance / v1.shape[0]
  return distance, np.sign(coef1 - coef2)
  
def l2norm(x):
  return np.sqrt((x **2).sum())
  
def l2normalize(x):
  n = l2norm(x)
  if n == 0:
    return x
  else:
    return x / n
  
def check_domain(x):
  """Ensure that x is in [-1, 1]
     before computing arccos(x)
  """
  return np.min((1., np.max((-1., x))))
  
def opening(w, normalize = False):
  """ The opening of the polyhedral cone with face normal to w
  """  
  wunit = l2normalize(w)
  iunit = l2normalize(np.ones_like(w))
  
  dot = np.dot(wunit.T, iunit).item()
  dot = check_domain(dot)
  
  alpha = 0.5 * np.pi - np.arccos(dot)
  if dot <= 0: # alpha in (0, -pi/2)
    alpha = -1. * alpha  
  if normalize:
    alpha = (alpha + np.pi) / (2. * np.pi)
  return alpha
  
def rotation2(w1, w2, alpha1, alpha2, rotations_dict=None):
  """Compute the angle of the rotation that sends the
     polyhedral cone of w_i to w_j where i = argmax_1,2 (alpha1, alpha2)
  """  
  # order vectors by the corresponding angle
  w1, w2 = (w1, w2) if alpha1 > alpha2 else (w2, w1)
  
  # shear w1 by alpha2 - alpha1
  diff = np.abs(alpha2 - alpha1)
  
  # create Toepliz matrix
  W1 = make_toepliz(w1)
  
  if diff > 0:
    W1 = shear(W1, diff, rotations_dict)
  
  #normalize
  w2 = l2normalize(w2)
  
  for row in range(W1.shape[0]):
    W1[row] = l2normalize(W1[row])
  
  angles = np.arccos(np.dot(W1, w2))  
  return np.min(angles)

def make_toepliz(w):
  """Make a Toepliz matrix with w as the first row
  """
  dim = w.shape[0]
  W = np.zeros((dim,dim))
  for j in range(dim):
    W[j,:] = np.roll(w.T,j,axis=1) 
  
  return W
  
def shear(w, angle, rotations_dict=None):
  """Shear the hyperplane normal to w along the
     identity line by angle
  """
  
  if angle == 0.:
    shear_factor = 0.
  else:
    shear_factor = 1. / np.tan(angle)
  
  M = make_transform(shear_factor, w.shape[0], rotations_dict)
  
  return np.dot(M,w)
  
def make_transform(factor, dim, rotations_dict=None):
  """Compute the n-dimensional shear matrix
     that shears by factor along the identity line
  """
  if rotations_dict is None:
    rot = make_rotation_implicit(dim)
    rinv = np.linalg.inv(rot)
  else:
    rot, rinv = rotations_dict[dim]
    
  shear_m = np.eye(dim)
  shear_m[dim-1,0] = factor
  
  return np.dot(rinv,np.dot(shear_m, rot))
  
def make_rotation_implicit(dim):
    """Align the dim-dimensional identity line with
       the first axis of the coordinate system
    """
    prod = np.eye(dim)
    one = np.ones((dim,1), dtype=np.float)
    one = one / np.linalg.norm(one)
    
    for i in range(dim -1):
      rot = np.eye(dim)
      length = np.sqrt(one[i,0]**2 + one[i+1,0]**2)
      cos = one[i+1, 0] / length
      sin = one[i] / length
      
      rot[i:i+2,i:i+2] = np.array([[cos, - sin], [sin, cos]])
      
      one = np.dot(rot, one)
      prod = np.dot(rot, prod)
    return prod
  
def init_rotations(dim_list):
  """Return a dictionary {"dim": "[R, Rinv]"} of rotations R
     aligning the dim dimensional identity line with the first axis and the
     corresponding inverse rotation, for each dim in dim_list
  """
  rotations = {}
  for dim in dim_list:
    R = make_rotation_implicit(dim)
    Rinv = np.linalg.inv(R)
    rotations[dim] = [R, Rinv]
  return rotations

def make_sparse(param, input_size):
  """Take the 3x3x1 kernel and flattens it into
     the first row of the corresponding vectorization
  """
  
  result = np.zeros((input_size[0], input_size[1]))
  result[0:3,0:3] = param
  result = np.reshape(result, (input_size[0] * input_size[1], 1))
  return result

def pairwise_nesting(w1, b1, w2, b2, input_size, rotations_dict=None, skip_rotations=False):
  """Compute the mutual nesting between two filters
     given their weights and biases
     
     Returns the signed distance of vertices and opening
     which can be used to detect partial vs full nesting
  """
  w1 = make_sparse(w1, input_size)
  w2 = make_sparse(w2, input_size)
  
  dist, sign = distance(w1, b1, w2, b2)
  
  alpha1 = opening(w1)
  alpha2 = opening(w2)
  
  if skip_rotations:
    rot = 0.
  else:
    rot = rotation2(w1, w2, alpha1, alpha2)
  
  # normalize shear angle
  alpha1 = 2. * alpha1 / np.pi
  alpha2 = 2. * alpha2 / np.pi
  
  rot = 2. * rot / np.pi
  
  return (sign*dist, alpha1 - alpha2, rot)
  
if __name__ == '__main__':
   
  w1 = np.random.randn(3,3)
  b1 = np.random.rand(1) * 3
  w2 = np.random.randn(3,3)
  b2 = np.random.rand(1) * 3
  input_size = (32, 32)
  
  result = pairwise_nesting(w1, b1, w2, b2, input_size)
  print(result)
  
