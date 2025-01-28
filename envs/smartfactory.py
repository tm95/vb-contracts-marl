import gym
import gym.spaces
from collections import namedtuple
import numpy as np
import Box2D
from Box2D import b2PolygonShape, b2FixtureDef, b2TestOverlap, b2Transform, b2Rot, b2ChainShape
import copy
import json
import itertools


class Agent:
    def __init__(self, world, env, index, position, task):
        self.world = world
        self.env = env
        self.index = index
        self.old_pos = position
        self.current_pos = position
        self.agent_vertices = [(-1, -1), (0, -1), (0, 0), (-1, 0)]
        self.body = self.world.CreateDynamicBody(position=position,
                                                 angle=0,
                                                 angularDamping=0.6,
                                                 linearDamping=3.0,
                                                 shapes=[b2PolygonShape(vertices=self.agent_vertices)],
                                                 shapeFixture=b2FixtureDef(density=0.0))
        self.signalling = False
        self.task = task
        self.episode_debts = 0
        self.done = False

    def process_task(self):
        x = self.body.transform.position.x
        y = self.body.transform.position.y

        task_index = -1
        if [x, y] in self.env.goal_positions:
            index_machine = self.env.goal_positions.index([x, y])
            machine = self.env.goals[index_machine]
            if machine.typ in self.task and machine.inactive <= 0:
                machine.inactive = self.env.nb_steps_machine_inactive
                index_task = self.task.index(machine.typ)
                self.task[index_task] = -1
                task_index = index_task

        return task_index

    def process_task_sequential(self):
        x = self.body.transform.position.x
        y = self.body.transform.position.y

        task_index = -1

        if [x, y] in self.env.goal_positions:

            index_machine = self.env.goal_positions.index([x, y])
            machine = self.env.goals[index_machine]

            if machine.typ == self.task[0] and machine.inactive <= 0:
                machine.inactive = self.env.nb_steps_machine_inactive
                del self.task[0]
                task_index = 0

        return task_index

    def tasks_finished(self):
        return all(i == -1 for i in self.task)

    def set_signalling(self, action):
        self.signalling = False

        if action[3] == 1:
            self.signalling = True

    def reset(self, position, task):
        self.body.position = position
        self.old_pos = position
        self.current_pos = position

        self.task = task
        self.done = False
        self.episode_debts = 0


class GridCell:

    def __init__(self, env, index, position, typ):
        self.index = index
        self.vertices = [(-1, -1), (0, -1), (0, 0), (-1, 0)]
        self.shape = b2PolygonShape(vertices=self.vertices)
        self.position = position
        self.typ = typ
        self.inactive = 0

    def reset(self, env, index):
        self.inactive = 0


class Smartfactory(gym.Env):
    State = namedtuple('Smartfactory', 'agent_states')

    def __init__(self,
                 nb_agents,
                 field_width,
                 field_height,
                 rewards,
                 priorities,
                 step_penalties,
                 contracting=0,
                 nb_machine_types=2,
                 machine_types=[0, 1, 0],
                 nb_machines=3,
                 nb_steps_machine_inactive=10,
                 nb_tasks=3,
                 observation=0):
        """

        :rtype: observation
        """
        self.world = Box2D.b2World(gravity=(0, 0))
        if observation == 0:
            self.observation_space = gym.spaces.Box(0.0, 1.1, shape=(84, 84, 1))
        elif observation == 1:
            self.nb_channels = 9
            self.observation_space = gym.spaces.Box(0.0, 1.1, shape=(self.nb_channels, field_width, field_height))

        self.velocity_iterations = 6
        self.position_iterations = 2
        self.dt = 1.0 / 15
        self.agent_restitution = 0.5
        self.agent_density = 1.0

        with open('envs/actions.json', 'r') as f:
            actions_json = json.load(f)

        self.contracting = contracting
        if self.contracting == 0:
            self.actions = actions_json['no_contracting_actions']
        if self.contracting == 1:
            self.actions = actions_json['contracting_actions']
        self.contracting = contracting

        self.nb_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(self.nb_actions)
        self.nb_contracting_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(n=len(self.actions))

        self.info = {'setting': None,
                     'episode': None,
                     'return_a0': None,
                     'return_a1': None
                     }

        self.pixels_per_worldunit = 24
        self.obs_pixels_per_worldunit = 8
        self.field_width = field_width
        self.field_height = field_height
        self.edge_margin = .2

        # Agents
        self.nb_agents = nb_agents
        self.field_vertices = []
        self.edge_vertices = []
        self.agents = []
        self.market_agents = []
        self.food = []
        self.goals = []
        self.tasks = []
        self.task_positions = []

        self.rewards = rewards
        self.step_penalties = step_penalties
        self.contract = False

        self.nb_machine_types = nb_machine_types
        self.nb_machines = nb_machines
        self.nb_steps_machine_inactive = nb_steps_machine_inactive
        self.nb_tasks = nb_tasks
        self.debt_balances = []
        self.balance = np.zeros(self.nb_agents)
        self.greedy = []
        self.map = None
        self.machine_types = machine_types
        self.machines = []
        self.debt_balance_position = []
        self.current_contract = [False, False]

        self.priorities = priorities
        self.wall_indices = []
        self.possible_positions = [[(-(self.field_width-(self.field_width/2)-1))+column,
                                    (self.field_height/2)-row] for row in range(self.field_width)
                                   for column in range(self.field_height)]
        self.wall_positions = []
        self.goal_positions = []

    def _create_field(self):
        self.field_vertices = [(-self.field_width/2, -self.field_height/2),
                               (self.field_width/2, -self.field_height/2),
                               (self.field_width/2, self.field_height/2),
                               (-self.field_width/2, self.field_height/2)]
        self.edge_vertices = [(x - self.edge_margin if x < 0 else x + self.edge_margin,
                               y - self.edge_margin if y < 0 else y + self.edge_margin)
                              for (x, y) in self.field_vertices]
        self.field = self.world.CreateBody(shapes=b2ChainShape(vertices=self.edge_vertices))

    def reset(self):
        self.world = Box2D.b2World(gravity=(0, 0))
        self.agents = []
        self.market_agents = []
        self.food = []
        self.goals = []
        self.tasks = []
        self.machines = []
        self.greedy = []
        tasks = []
        self.debt_balances = []

        np_indices = list(itertools.product([i for i in range(self.field_width)], [i for i in range(self.field_width)]))

        self.map = {'{}-{}'.format(pp[0], pp[1]): index for pp, index in zip(self.possible_positions, np_indices)}

        field_indices = [pos for pos in range(self.field_width*self.field_height)]
        wall_indices = []
        goal_indices = [0, self.field_width-1, (self.field_width*self.field_height) - self.field_width]
        self.wall_positions = [self.possible_positions[i] for i in wall_indices]
        self.goal_positions = [self.possible_positions[i] for i in goal_indices]
        spawning_positions = list(set(field_indices) - set(wall_indices) - set(goal_indices))

        spawning_indices = np.random.choice(spawning_positions, self.nb_agents, replace=False)

        for i in range(self.nb_agents):
            tasks.append(list(np.random.choice(self.machine_types, self.nb_tasks)))

        machines = [(m_pos, m_typ) for m_pos, m_typ in zip(self.goal_positions, self.machine_types)]

        for i in self.machine_types:
            self.machines.append([machine[0] for machine in machines if machine[1] == i])

        self.task_positions = [(-self.field_width/2 + (1 + (i * 2)),
                                -self.field_height/2 + -1) for i in range(self.nb_tasks)]

        self.debt_balance_position = [(-self.field_width/2 + 3, self.field_height/2 + 2)]

        for i in range(self.nb_agents):
            agent = Agent(world=self.world,
                          env=self,
                          index=i,
                          position=self.possible_positions[spawning_indices[i]],
                          task=tasks[i])
            self.agents.append(agent)

        for i, goal_pos in enumerate(self.goal_positions):
            self.goals.append(GridCell(env=self,
                                       index=i,
                                       position=goal_pos,
                                       typ=self.machine_types[i]))

        for i, task_pos in enumerate(self.task_positions):
            self.tasks.append(GridCell(env=self,
                                       index=i,
                                       position=task_pos,
                                       typ='task-{}'.format(i)))

        for i, debt_balance_pos in enumerate(self.debt_balance_position):
            self.debt_balances.append(GridCell(env=self,
                                               index=0,
                                               position=debt_balance_pos,
                                               typ='debt_balance'))

        self._create_field()

        return self.observation

    def step(self, actions):
        """
        :param actions: the list of agent actions
        :type actions: list
        """
        info = copy.deepcopy(self.info)
        rewards = np.zeros(self.nb_agents)

        joint_actions = []
        for i_ag in range(self.nb_agents):
            joint_actions.append(self.actions[actions[i_ag]])
        actions = joint_actions

        if any([agent.done for agent in self.agents]):
            for agent in self.agents:
                agent.episode_debts = 0

        queue = np.random.choice([i for i in range(self.nb_agents)], self.nb_agents, replace=False)
        for i in queue:
            agent = self.agents[i]
            if not agent.done:
                self.set_position(agent, actions[agent.index])

                if self.priorities[i]:
                    rewards[i] -= self.step_penalties[0]
                else:
                    rewards[i] -= self.step_penalties[1]

                if agent.process_task_sequential() >= 0:
                    if self.priorities[i]:
                        rewards[i] += self.rewards[0]
                    else:
                        rewards[i] += self.rewards[1]

                if agent.tasks_finished():
                    agent.done = True

        self.process_machines()

        return self.observation, rewards, [agent.done for agent in self.agents], info

    @property
    def observation(self):
        """
        OpenAI Gym Observation
        :return:
            List of observations
        """
        observations = []
        for i_agent, agent in enumerate(self.agents):
            observation = self.observation_one_hot(i_agent)
            observations.append(observation)

        return observations

    def set_position(self, agent, action):

        agent.old_pos = agent.current_pos

        new_pos = [agent.body.transform.position.x + (1.0 * action[0]),
                   agent.body.transform.position.y + (1.0 * action[1])]

        if new_pos not in self.wall_positions:
            if action[1] == 1.0:  # up
                if new_pos[1] <= self.field_vertices[2][1]:
                    agent.body.transform = (new_pos, 0)
                    agent.current_pos = new_pos
            if action[1] == -1.0:  # down
                if new_pos[1] > self.field_vertices[0][1]:
                    agent.body.transform = (new_pos, 0)
                    agent.current_pos = new_pos
            if action[0] == -1.0:  # left
                if new_pos[0] > self.field_vertices[0][0]:
                    agent.body.transform = (new_pos, 0)
                    agent.current_pos = new_pos
            if action[0] == 1.0:  # right
                if new_pos[0] <= self.field_vertices[1][0]:
                    agent.body.transform = (new_pos, 0)
                    agent.current_pos = new_pos

    def process_machines(self):
        for i_machine, machine in enumerate(self.goals):
            if machine.inactive > 0:
                machine.inactive -= 1

    def observation_one_hot(self, agent_id):

        channels = self.nb_channels
        observation = np.zeros((channels, self.field_width, self.field_height))

        c_active_machines = 0
        c_pos_self = 1
        c_task_prio = 2
        c_tasks = [i+3 for i in range(self.nb_tasks)]

        c_pos_other = 6
        c_task_0_other = 7

        c_other_agent_done = 8

        for g in self.goals:
            x_m = int(self.map['{}-{}'.format(g.position[0], g.position[1])][0])
            y_m = int(self.map['{}-{}'.format(g.position[0], g.position[1])][1])

            if g.inactive <= 0:
                observation[c_active_machines][x_m][y_m] += 1

        x_a_raw = self.agents[agent_id].body.transform.position.x
        y_a_raw = self.agents[agent_id].body.transform.position.y
        x_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][0])
        y_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][1])
        observation[c_pos_self][x_a][y_a] += (agent_id + 1)

        for i_task, task in enumerate(self.agents[agent_id].task):
            for x_task_raw, y_task_raw in self.machines[task]:
                x_task = int(self.map['{}-{}'.format(x_task_raw, y_task_raw)][0])
                y_task = int(self.map['{}-{}'.format(x_task_raw, y_task_raw)][1])

                observation[c_tasks[i_task]][x_task][y_task] += 1

        # other agents
        for agent in range(self.nb_agents):
            if (len(self.agents) > 1) and agent != agent_id:
                if self.agents[agent].done:
                    observation[c_other_agent_done] += 1
                else:

                    x_a_raw = self.agents[agent].body.transform.position.x
                    y_a_raw = self.agents[agent].body.transform.position.y
                    x_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][0])
                    y_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][1])
                    observation[c_pos_other][x_a][y_a] += (agent + 1)

                    if len(self.agents[agent].task) > 0:
                        for x_task_other_raw, y_task_other_raw in self.machines[self.agents[agent].task[0]]:
                            x_task_other = int(self.map['{}-{}'.format(x_task_other_raw, y_task_other_raw)][0])
                            y_task_other = int(self.map['{}-{}'.format(x_task_other_raw, y_task_other_raw)][1])
                            observation[c_task_0_other][x_task_other][y_task_other] += 1

        observation[c_task_prio] += self.priorities[agent_id]

        return observation

    @staticmethod
    def overlaps_checkpoint(checkpoint, agent):
        return b2TestOverlap(
            checkpoint.shape, 0,
            agent.body.fixtures[0].shape, 0, b2Transform(checkpoint.position, b2Rot(0.0)),
            agent.body.transform)

    @staticmethod
    def distance_agents(agent_1, agent_2):
        dist_x = agent_1.body.transform.position.x - agent_2.body.transform.position.x
        dist_y = agent_1.body.transform.position.y - agent_2.body.transform.position.y
        return np.sqrt((dist_x*dist_x) + (dist_y*dist_y))

    @staticmethod
    def agent_collision(agent_1, agent_2):
        collision = False
        if agent_1.old_pos == agent_2.current_pos and agent_1.current_pos == agent_2.old_pos:
            collision = True
        if agent_1.current_pos == agent_2.current_pos:
            collision = True
        return collision


def make_smart_factory(params):

    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       priorities=params.priorities,
                       step_penalties=params.step_penalties,
                       nb_machine_types=params.nb_machine_types,
                       machine_types=params.machine_types,
                       nb_machines=params.nb_machines,
                       nb_steps_machine_inactive=params.nb_steps_machine_inactive,
                       nb_tasks=params.nb_tasks,
                       contracting=params.contracting,
                       observation=1
                       )
    return env
