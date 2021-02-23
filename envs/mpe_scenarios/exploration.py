import numpy as np
from multiagent.core import World, Agent, Landmark, Block, Point
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        #num_landmarks = 2
        num_blocks = 5
        #num_points = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        '''
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        '''
        world.blocks = [Block() for i in range(num_blocks)]
        for i, blocks in enumerate(world.blocks):
            blocks.name = 'block %d' % i
            blocks.collide = True
            blocks.movable = False
            blocks.size = 0.2
            blocks.boundary = False
        #world.landmarks += world.blocks
        '''
        world.points = [Point() for i in range(num_points)]
        for i, point in enumerate(world.points):
            point.name = 'point %d' % i
            point.collide = True
            point.movable = False
            point.size = 0.3
            #point.boundary = False
        '''
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        #input()
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        '''
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        '''
        # random properties for blocks
        for i, block in enumerate(world.blocks):
            block.color = np.array([0.15, 0.15, 0.65])

        '''
        for i, point in enumerate(world.points):
            point.color = np.array([0.15, 0.15, 0.65])
        '''

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        '''
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        '''
        for i, block in enumerate(world.blocks):
            block.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            block.state.p_vel = np.zeros(world.dim_p)

        '''
        for i, point in enumerate(world.points):
            point.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            point.state.p_vel = np.zeros(world.dim_p)
        '''


    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_rate = 0
        min_dists = 0
        '''
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        '''
        for p in world.points:
            p.state.p_pos += 1


        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_rate)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_collision_point(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent1.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        collisions = 0
        collisions_with_blocks = 0
        '''
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        '''

        '''
        #agents are awarded based on how far they are from the nearest point
        for l in world.points:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew += min(dists)
        '''

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 2
                    collisions += 1
            for a in world.points:
                if self.is_collision_point(a, agent):
                    rew -= 1
            for a in world.blocks:
                if self.is_collision_point(a, agent):
                    rew -= 3
                    collisions_with_blocks += 1

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        '''
        #agents are penalized for colide with any point, so that they trend to explore bigger space
        for a in world.points:
            if self.is_collision_point(a,agent):
                rew -=1
            #if not self.is_collision_point(a,agent):
             #   rew +=1
        '''
        return rew,collisions,collisions_with_blocks

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
