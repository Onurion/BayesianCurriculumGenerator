import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import rendering
import torch


################################################################################
class Zone():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.r = 0.0
        self.status = 0

class Rocket():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.psi = 0.0
        self.life = 0.0
        self.fired = 0.0

class Agent():
    def __init__(self):
        self.x = -30.0
        self.y = -30.0
        self.v = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.psi = 0.0
        self.psiDot = 0.0
        self.angleBetween = None

class Target():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
        self.claimed = False

class Enemy():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.v = 4.0
        self.psi = 0.0
        self.strategy = 0
################################################################################
class Aerorival(gym.Env):
    def __init__(self, n_zones:int, agent_init_position:list, target_distance_coeff:float, target_positions:list, enemy_velocity:float, zone_prob:float, rocket_velocity:float, seed:int, visualization:bool, difficulty:str="hard"):
        super(Aerorival, self).__init__()
        self.visualization = visualization
        self.seed()
        self.numberOfSucRuns = 0
        self.numberOfSemiSucRuns = 0
        self.map_lim = 50
        self.position_threshold = 2.5
        self.max_velocity = 10.0
        self.view_size = 500
        self.n_zones = n_zones #self.np_random.randint(1,5) #n_zones
        self.n_targets = len(target_positions)
        self.agent_init_position = agent_init_position
        self.enemy_velocity = enemy_velocity
        self.rocket_velocity = rocket_velocity
        self.zone_prob = zone_prob
        self.difficulty = difficulty

        # STATES
        # [difference_x, difference_y, agent_v, enemy_v, difference_v, psi_agent, psi_enemy, difference_angle, strategy_enemy] - Enemy
        # [difference_x, difference_y, difference_angle] - Targets
        # [difference_x, difference_y, difference_angle, radius, status] - Zones
        # [difference_x, difference_y, difference_angle, v, fired] - Rockets
        
        N_obs = 9 + self.n_targets * 3 + self.n_zones * 10

        High = np.ones(N_obs, dtype=np.float32)
        Low = -1 * np.ones(N_obs, dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=Low,
            high=High,
            shape=(N_obs,),
            dtype=np.float32
        )

        #Map max/min into acceleration = deltaV[-1,1] where 1<V<5
        #psidot [-pi/3, pi/3]
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1,  1]), shape=(2,),
            dtype=np.float32
        ) 


        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        torch.manual_seed(seed)

        #Define agent and target object
        self.agent = Agent()
        self.targets = [Target() for _ in range(self.n_targets)]
        self.enemy = Enemy()

        self.agent.angleBetween = np.zeros(self.n_targets)

        #Other episodic variables
        self.viewer = None
        self.dt = .05
        # self.obstacleList = []
        self.done = False
        self.iteration = 0
        self.reward = 0

        self.max_iterations = 1000
        self.inside = [False]*self.n_zones

        #Initialize Zones and Rockets
        self.numberOfRockets = self.n_zones      
        self.zones = [Zone() for _ in range(self.n_zones)]
        self.rockets = [Rocket() for _ in range(self.numberOfRockets)]

        

        self.distance_list = []
        self.angle_list = []
        # target_x = self.map_lim - 10

        # interval = 2 * self.map_lim / (self.n_targets + 1)
        # target_y_loc = [-self.map_lim + (i + 1) * interval for i in range(self.n_targets)]

        for target_x, target_y in target_positions:
            angle = angle_normalize(np.arctan2(target_y-self.agent_init_position[1] , target_x - self.agent_init_position[0]))
            dist = np.sqrt((target_y - self.agent_init_position[1])**2 + (target_x - self.agent_init_position[0])**2)
            self.distance_list.append(dist * target_distance_coeff)
            self.angle_list.append(angle)

    
    ########################################
    def checkCloseness(self, zones, unit):
        for i in range(self.n_zones):
            if np.sqrt((unit.x-zones[i].x)**2 + (unit.y-zones[i].y)**2) <= zones[i].r:
                return True
            else:
                continue
        return False
    
    def isInside(self, zone, unit, margin=0.0):
        return True if np.sqrt((unit.x-zone.x)**2 + (unit.y-zone.y)**2) <= (zone.r + margin) else False

    ########################################
    def initializeZones(self):
        zone_edge = self.map_lim - 5
        radius = 2 * zone_edge / (2 * self.n_zones)
        if self.difficulty == "easy":
            radius = radius / 2.0

        for i in range(self.n_zones):
            self.zones[i].x = 0 #0 if i==0 else 0
            if self.difficulty == "easy":
                self.zones[i].y = 0
            elif self.difficulty == "hard":
                self.zones[i].y = zone_edge - (1+i*2)*radius #round(self.np_random.uniform(low=zone_y_loc[i]-2.5, high=zone_y_loc[i] + 2.5),2) #25 if i==0 else -25
            self.zones[i].r = radius #round(self.np_random.uniform(low=radius-5.0, high=radius+5.0),2) #round(self.np_random.uniform(low=20.0, high=25.0),2)#
            self.zones[i].status = 1#self.np_random.choice([0, 1], p=[1-self.zone_prob, self.zone_prob])
           
    ########################################
    def initializeTarget(self):
        # tooClose = True
        # while tooClose:
        interval = 2 * self.map_lim / (self.n_targets + 1)
        target_y_loc = [-self.map_lim + (i + 1) * interval for i in range(self.n_targets)]
        target_x = self.map_lim - 10
        

        for i in range(self.n_targets):
            self.targets[i].x = self.distance_list[i] * np.cos(self.angle_list[i]) + self.agent_init_position[0] #target_x  #40 if i==0 else 40
            self.targets[i].y = self.distance_list[i] * np.sin(self.angle_list[i]) + self.agent_init_position[1] #target_y_loc[i] #30 if i==0 else -30
            self.targets[i].psi = 0
          
    ########################################
    def initializeAgent(self):
        margin = np.maximum(np.abs(self.agent_init_position[0] / 3), np.abs(self.agent_init_position[1] / 3))
        self.agent.x = self.np_random.uniform(low=self.agent_init_position[0] - margin, high= self.agent_init_position[0] + margin)
        self.agent.y = self.np_random.uniform(low=self.agent_init_position[1] - margin, high= self.agent_init_position[1] + margin)
        self.agent.psi = self.np_random.uniform(low=-np.pi, high=np.pi)
        self.agent.psiDot = 0.0
        self.agent.v = self.np_random.uniform(low=0.0, high=2.0)
        self.agent.vx = self.agent.v * np.cos(self.agent.psi)
        self.agent.vy = self.agent.v * np.sin(self.agent.psi)
    ########################################
    def initializeEnemy(self):
        self.enemy.strategy = self.np_random.integers(0,self.n_targets)

        self.enemy.x =  self.np_random.uniform(low=-40, high=-20)
        self.enemy.y = self.np_random.uniform(low=20, high=40)
        self.enemy.psi = self.np_random.uniform(low=-np.pi, high=np.pi)
        self.enemy.v = self.enemy_velocity#self.np_random.uniform(low=1.0, high=3.0)
    ########################################
    def initializeRockets(self):
        for i in range(self.numberOfRockets):
            self.rockets[i].x = self.zones[i].x
            self.rockets[i].y = self.zones[i].y
            self.rockets[i].v = 0.0
            self.rockets[i].psi = 0.0
            self.rockets[i].fired = 0.0
            self.rockets[i].life = 0.0
    ########################################
    def UpdateRockets(self):
        done = False
        rocket_life_margin = 2.0 #life span outside the radius
        for i in range(self.numberOfRockets):
            #Check if agent inside any NF Zone
            self.inside[i] = self.isInside(self.zones[i], self.agent)

            #if outside of the map  or rocket reached end of life - respawn rocket and set fired status and velocity zero
            #life boundary can be adjusted !!!
            if (abs(self.rockets[i].x)>self.map_lim or abs(self.rockets[i].y)>self.map_lim or not self.isInside(self.zones[i], self.rockets[i], margin=rocket_life_margin)) and self.zones[i].status == 1:
                self.inside[i] = False
                self.rockets[i].v = 0.0
                self.rockets[i].fired = 0.0
                self.rockets[i].x = self.zones[i].x
                self.rockets[i].y = self.zones[i].y
                self.rockets[i].psi = 0.0
                self.rockets[i].life = 0
            
            #Fire rockets if not already fired 
            elif(self.inside[i] or self.rockets[i].fired == 1.0) and self.zones[i].status == 1:
                self.rockets[i].fired = 1.0
                self.rockets[i].v = self.rocket_velocity
                self.rockets[i].psi = angle_normalize(np.arctan2(self.agent.y-self.rockets[i].y , self.agent.x-self.rockets[i].x))
                self.rockets[i].x +=  self.rockets[i].v * np.cos(self.rockets[i].psi)*self.dt
                self.rockets[i].y +=  self.rockets[i].v * np.sin(self.rockets[i].psi)*self.dt
                self.rockets[i].life += 1


            if np.sqrt((self.agent.x-self.rockets[i].x)**2 + (self.agent.y-self.rockets[i].y)**2) < self.position_threshold:
                # collision with the agent
                done = True

        return done
    ########################################
    def UpdateEnemy(self):   
        #Make enemy go to its target
        done = False

        target_position = self.targets[self.enemy.strategy]
        pos_diff_target = np.sqrt((self.enemy.x-target_position.x)**2 + (self.enemy.y-target_position.y)**2)
        pos_diff_agent = np.sqrt((self.enemy.x-self.agent.x)**2 + (self.enemy.y-self.agent.y)**2)

        if  pos_diff_target < self.position_threshold or pos_diff_agent < self.position_threshold:
            self.enemy.v = 0.0
            done = True

            
        self.enemy.psi = angle_normalize(np.arctan2(target_position.y-self.enemy.y , target_position.x-self.enemy.x))
        self.enemy.x +=  self.enemy.v * np.cos(self.enemy.psi)*self.dt
        self.enemy.y +=  self.enemy.v * np.sin(self.enemy.psi)*self.dt
        return done
    ########################################
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    ########################################

    def step(self, action):
        action = np.clip(action, [-1.,-1.],
                                [1.,1.])
        # input("Press Enter to continue...")

        #Scale  and update agent actions
        acceleration = action[0] * 2 # v_dot - action scaled between [-2, 2]
        self.agent.v = np.clip(self.agent.v+acceleration,0,self.max_velocity) #acceleration[-1,+1] - velocity scaled between [0,10]
        self.agent.psiDot = action[1]*(np.pi/2) #psi_dot - action scaled between [-pi/2, pi/2]
        
        #Update agent parameters
        self.agent.psi = angle_normalize(self.agent.psi + self.agent.psiDot*self.dt)
        self.agent.vx = self.agent.v * np.cos(self.agent.psi)
        self.agent.vy = self.agent.v * np.sin(self.agent.psi)

        self.agent.x = self.agent.x + self.agent.vx * self.dt
        self.agent.y = self.agent.y + self.agent.vy * self.dt
        
        collision_with_rocket = self.UpdateRockets()
        collision_with_enemy = self.UpdateEnemy()
        target_position = self.targets[self.enemy.strategy]
        self.enemy.psi = angle_normalize(np.arctan2(target_position.y-self.enemy.y , target_position.x-self.enemy.x))

        self.iteration += 1

        if abs(self.agent.x) >= self.map_lim or abs(self.agent.y) >= self.map_lim or collision_with_rocket or collision_with_enemy or self.iteration >= self.max_iterations:
            self.done = True
            self.reward = -1.0
        elif np.sqrt((self.agent.x-target_position.x)**2 + (self.agent.y-target_position.y)**2) < self.position_threshold:
            self.done = True
            self.reward = 1.0 - self.iteration * 0.0005
        
        # STATES
        # [difference_x, difference_y, agent_v, enemy_v, difference_v psi_agent, psi_enemy, difference_angle, strategy_enemy] - Enemy
        # [difference_x, difference_y, difference_angle] - Targets
        # [difference_x, difference_y, difference_angle, radius, status] - Zones
        # [difference_x, difference_y, difference_angle, v, fired] - Rockets

        state1 = [(self.agent.x-self.enemy.x) / (2*self.map_lim), (self.agent.y-self.enemy.y) / (2*self.map_lim), self.agent.v / self.max_velocity, self.enemy.v / self.max_velocity, (self.agent.v - self.enemy.v) / self.max_velocity,
                  self.agent.psi / np.pi , self.enemy.psi / np.pi,  angle_normalize(self.agent.psi - self.enemy.psi) / (2*np.pi), self.enemy.strategy]
        
        for i in range(self.n_targets):
            state1.append([(self.agent.x-self.targets[i].x) / (2*self.map_lim), (self.agent.y-self.targets[i].y) / (2*self.map_lim), 
                           angle_normalize(np.arctan2(self.targets[i].y-self.agent.y, self.targets[i].x-self.agent.x)) / (2*np.pi)])
        
        for i in range(self.n_zones):
            state1.append([(self.agent.x-self.zones[i].x) / (2*self.map_lim), (self.agent.y-self.zones[i].y) / (2*self.map_lim), 
                           angle_normalize(np.arctan2(self.zones[i].y-self.agent.y, self.zones[i].x-self.agent.x)) / (2*np.pi), self.zones[i].r / self.map_lim, self.zones[i].status])
            
            state1.append([(self.agent.x-self.rockets[i].x) / (2*self.map_lim), (self.agent.y-self.rockets[i].y)/(2*self.map_lim), 
                           angle_normalize(np.arctan2(self.rockets[i].y-self.agent.y, self.rockets[i].x-self.agent.x)) / (2*np.pi), self.rockets[i].v / self.max_velocity, self.rockets[i].fired])

        self.state = np.concatenate([np.array(item) if isinstance(item, list) else np.array([item]) for item in state1])


        if self.visualization:
            img = self.render()  
            if self.done:
                self.close()

        return self.state, self.reward, self.done, {}

    ########################################
    def reset(self):
        #Reset zone parameters
        self.initializeZones()
        #Reset target location
        self.initializeTarget()
        #Reset enemy parameters
        self.initializeEnemy()
        #Reset agent parameters
        self.initializeAgent()
        
        #Reset rocket parameters
        self.initializeRockets()

        #Reset transforms
        self.nfz = []
        self.nfz_transform = []
        self.rocket = []
        self.rocket_transform = []
        self.target = []
        self.target_transform = []
        self.inside = [False] * self.n_zones

        # STATES
        # [difference_x, difference_y, agent_v, enemy_v, difference_v psi_agent, psi_enemy, difference_angle, strategy_enemy] - Enemy
        # [difference_x, difference_y, difference_angle] - Targets
        # [difference_x, difference_y, difference_angle, radius, status] - Zones
        # [difference_x, difference_y, difference_angle, v, fired] - Rockets

        state1 = [(self.agent.x-self.enemy.x) / (2*self.map_lim), (self.agent.y-self.enemy.y) / (2*self.map_lim), self.agent.v / self.max_velocity, self.enemy.v / self.max_velocity, (self.agent.v - self.enemy.v) / self.max_velocity,
                  self.agent.psi / np.pi , self.enemy.psi / np.pi,  angle_normalize(self.agent.psi - self.enemy.psi) / np.pi, self.enemy.strategy]
        
        for i in range(self.n_targets):
            state1.append([(self.agent.x-self.targets[i].x) / (2*self.map_lim), (self.agent.y-self.targets[i].y) / (2*self.map_lim), 
                           angle_normalize(np.arctan2(self.targets[i].y-self.agent.y, self.targets[i].x-self.agent.x)) / np.pi])
        
        for i in range(self.n_zones):
            state1.append([(self.agent.x-self.zones[i].x) / (2*self.map_lim), (self.agent.y-self.zones[i].y) / (2*self.map_lim), 
                           angle_normalize(np.arctan2(self.zones[i].y-self.agent.y, self.zones[i].x-self.agent.x)) / np.pi, self.zones[i].r / self.map_lim, self.zones[i].status])
            
            state1.append([(self.agent.x-self.rockets[i].x) / (2*self.map_lim), (self.agent.y-self.rockets[i].y)/(2*self.map_lim), 
                           angle_normalize(np.arctan2(self.rockets[i].y-self.agent.y, self.rockets[i].x-self.agent.x)) / np.pi, self.rockets[i].v / self.max_velocity, self.rockets[i].fired])

        self.state = np.concatenate([np.array(item) if isinstance(item, list) else np.array([item]) for item in state1])

        
        self.done = False
        self.iteration = 0
        self.reward = 0

        return self.state
    ########################################
    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.view_size, self.view_size)
            self.viewer.set_bounds(-self.map_lim, self.map_lim, -self.map_lim, self.map_lim)

            # Add plane pic
            fname = "./assets/plane.png"
            self.plane_transform = rendering.Transform()
            self.plane = rendering.Image(fname, 5., 5.)
            self.plane.add_attr(self.plane_transform)

            # Add enemy plane pic
            fname1 = "./assets/plane2.png"
            self.plane2_transform = rendering.Transform()
            self.plane2 = rendering.Image(fname1, 5., 5.)
            self.plane2.add_attr(self.plane2_transform)

        if not self.rocket:
            # Added rocket pic
            fname3 = "./assets/rocket.png"
            for i in range(self.numberOfRockets):
                self.rocket_transform.append(rendering.Transform())
                self.rocket.append(rendering.Image(fname3, 5., 5.))
                self.rocket[i].add_attr(self.rocket_transform[i])
        
        if not self.target:
            # Added target pic
            fname2 = "./assets/target.png"
            for i in range(self.n_targets):
                self.target_transform.append(rendering.Transform())
                self.target.append(rendering.Image(fname2, 5., 5.))
                self.target[i].add_attr(self.target_transform[i])
        
        if not self.nfz:  
            for i in range(self.n_zones):
                # Added zone pic
                fname4 = "./assets/nfz.png" if self.zones[i].status else "./assets/inactive_nfz.png"
                self.nfz_transform.append(rendering.Transform())
                self.nfz.append(rendering.Image(fname4, self.zones[i].r*2, self.zones[i].r*2))
                self.nfz[i].add_attr(self.nfz_transform[i])

        for i in range(self.n_targets):
            # Add nfz position and orientation
            self.viewer.add_onetime(self.target[i])
            self.target_transform[i].set_translation(self.targets[i].x, self.targets[i].y)
            self.target_transform[i].set_rotation(0.0)

        for i in range(self.n_zones):
            # Add nfz position and orientation
            self.viewer.add_onetime(self.nfz[i])
            self.nfz_transform[i].set_translation(self.zones[i].x, self.zones[i].y)
            self.nfz_transform[i].set_rotation(0.0)

        for i in range(self.numberOfRockets):
            if self.rockets[i].fired == 1:
                # Add target position and orientation
                self.viewer.add_onetime(self.rocket[i])
                self.rocket_transform[i].set_translation(self.rockets[i].x, self.rockets[i].y)
                self.rocket_transform[i].set_rotation(self.rockets[i].psi)

        # Add plane position and orientation
        self.viewer.add_onetime(self.plane)
        self.plane_transform.set_translation(self.agent.x, self.agent.y)
        self.plane_transform.set_rotation(self.agent.psi)

        # Add plane position and orientation
        self.viewer.add_onetime(self.plane2)
        self.plane2_transform.set_translation(self.enemy.x, self.enemy.y)
        self.plane2_transform.set_rotation(self.enemy.psi)


        return self.viewer.render(return_rgb_array=mode == 'human')

    ########################################
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

################################################################################
def angle_normalize(angle):
    while(angle >= np.pi):
        angle -= 2 * np.pi
    while(angle < -np.pi):
        angle += 2 * np.pi
    return angle


