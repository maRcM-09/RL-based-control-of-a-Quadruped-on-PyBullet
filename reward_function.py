def des_pitch_quat(self, des_pitch):
    half_angle = des_pitch/2
    qx =0
    qy = np.sin(half_angle)
    qz =0
    qw = np.cos(half_angle)

    return np.array([qx, qy, qz, qw])

def _reward_lr_course(self):
  """ Implement your reward function here. How will you improve upon the above? """
  # Base position
  # self.init_pos= self.robot._GetDefaultInitPosition()
  

  base_pos = self.robot.GetBasePosition()
  x = base_pos[0]
  z = base_pos[2]

  base_orient = self.robot.GetBaseOrientationRollPitchYaw()
  yaw = base_orient[2]

  delta_z = (z - self.prev_z) 
  delta_x = 1.0*(x - self.prev_x)/self._time_step

  self.prev_x = x
  self.prev_z = z

  z_reward = 5*delta_z if delta_z>0 else -0.05
  num_valid_contacts, _, _, _ = self.robot.GetContactInfo()
  contact_penalty = -1 if num_valid_contacts==4 else 0
  contact_reward = 0.5 * (4 - num_valid_contacts)

  yaw_penalty = -0.2 * np.abs(yaw)


  des_pitch = np.arctan2(self.stair_height+0.05, self.stair_width+0.05)
  des_quat = self.des_pitch_quat(des_pitch)
  base_quat = self.robot.GetBaseOrientation()
  stability_penalty = - 0.1 * np.linalg.norm(base_quat - des_quat) #allow slight tilting for climbing

  energy_reward = 0 
  for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
    energy_reward += np.abs(np.dot(tau,vel)) * self._time_step

  # c1,c2,c3,c4 = self.robot.GetContactInfo()
  # c3 =self.robot.GetBaseAngularVelocity()
  # print(c3)
  
  reward = delta_x \
        +  z_reward \
        +  stability_penalty \
        + yaw_penalty \
        - 0.005 * energy_reward \
        + contact_penalty \
        + contact_reward \
        
        
  return max(reward,0)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def _reward_lr_course(self):
  """ Implement your reward function here. How will you improve upon the above? """
  base_pos = self.robot.GetBasePosition()

  # define current stairs
  current_stair = self.current_stair()
    
  return 0
def _reward_stair(self,current_stair,z_robot):
  _,pitch,_ = self.robot.GetBaseOrientationRollPitchYaw()
  if not current_stair:
    return 0
  else:
    norm_pitch = np.tanh(pitch)
    _, z_stair, direction = current_stair[0]
    if direction == 'up':
      return 0
    if direction == 'down':
      return 0
 
      

  def tabular_stair(self):
    stairs = []
    curr_x = 0
    num_stair = self.num_stair
    width_stair = self.width_stair
    for i in range(num_stair):
      if i < num_stair/2:
        direction = 'up'
      else:
        direction = 'down'
      
      stairs.append((curr_x,direction))
      curr_x += width_stair
    return stairs
  
  def current_stair(self):
    tab_stair = self.tabular_stair()
    x = self.robot.GetBasePosition()[0]

    if x < tab_stair[0][0] or x > tab_stair[-1][0]:
      return []

    for i in range(1,len(tab_stair)):
      if tab_stair[i-1][0] <= x < tab_stair[i][0]:
        return  [tab_stair[i-1]]

    return []
" this code should be add " 
"   def add_stairs(self, num_stairs=12, stair_height=0.05, stair_width=0.25): ""
"""Add N stairs, with stair_height and stair_width. long so can't get around """
"    self._nb_stair = num_stairs  "
"    self._height_stair = stair_height  "
"    self._width_stair = stair_width  "