import training
import evaluation
from agents.dqn_agent import make_dqn_agent
from agents.a2c_agent import make_a2c_agent
from envs.smartfactory import make_smart_factory
from contracting.contracting import make_contract
from contracting.contracting_net import make_contracting_net
import json
from dotmap import DotMap
from datetime import datetime
import os
import neptune
import sys


def train(nb_steps_machine_inactive, nb_contracting_steps, seed, num_agents, gamma):

    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    params = ('params-{}.json'.format(0))
    with open(params, 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    for training_mode in [0, 1]:  # 0=Training, 1=No_Training
        for agent_type in [0]:  # 0=DQN, 1=A2C
            for contracting in [0, 1]:  # 0=Not_Contracting, 1=Contracting
                params.nb_steps_machine_inactive = nb_steps_machine_inactive
                params.nb_contracting_steps = nb_contracting_steps
                params.contracting = contracting
                params.nb_agents = num_agents
                params.gamma = gamma

                log_dir = os.path.join(os.getcwd(), 'experiments',
                                       'experiments-{}'.format(exp_time),
                                       'setting-{}'.format(0),
                                       'inactive-{}'.format(params.nb_steps_machine_inactive),
                                       'contracting-steps-{}'.format(params.nb_contracting_steps))

                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                logger = neptune.init_run(monitoring_namespace = "monitoring")
                logger["parameters"] = params_json
                run_experiment(logger, params, log_dir, agent_type, training_mode, seed)


def run_experiment(logger, params, log_dir, agent_type, training_mode, seed):

    agents = []
    env = make_smart_factory(params)
    observation_shape = list(env.observation_space.shape)
    number_of_actions = env.nb_actions
    weights_dir = os.path.join(log_dir, 'agent-{}'.format(agent_type),
                               'contracting-{}'.format(params.contracting), 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    policy_nets = []
    if params.contracting == 1:
        for i in range(params.nb_agents):
            policy_net = make_dqn_agent(params, observation_shape, params.nb_actions_no_contracting, seed)
            policy_net.epsilon = 0.05
            policy_net.load_weights(os.path.join(log_dir, 'agent-{}'.format(0),
                                                 'contracting-{}'.format(0),
                                                 'weights', "weights-{}.pth".format(i)))
            policy_nets.append(policy_net)

    contracting_net = make_contracting_net(params, observation_shape)
    contract = make_contract(params, policy_nets, contracting_net)

    # make agents and load weights
    for i in range(env.nb_agents):
        if agent_type == 0:
            agent = make_dqn_agent(params, observation_shape, number_of_actions, seed)
        elif agent_type == 1:
            agent = make_a2c_agent(observation_shape, number_of_actions, seed)
        agents.append(agent)

    # train agent and save weights
    if training_mode == 0:
        training.train(env, contract, agents, params.nb_episodes,
                       params.nb_steps, params.contracting, logger)

        for i in range(env.nb_agents):
            agents[i].save_weights(os.path.join(weights_dir, "weights-{}.pth".format(i)))

        if params.contracting == 1:
            contracting_net.save_weights(os.path.join(weights_dir, "weights-contracting"))

    # evaluate agents and load weights
    if training_mode == 1:
        for i, agent in enumerate(agents):
            agent.epsilon = params.epsilon_min
            agent.load_weights(os.path.join(log_dir, 'agent-{}'.format(agent_type),
                                            'contracting-{}'.format(params.contracting),
                                            'weights', "weights-{}.pth".format(i)))
        if params.contracting == 1:
            contracting_net.epsilon = params.epsilon_min
            contracting_net.load_weights(os.path.join(weights_dir, "weights-contracting"))
        evaluation.evaluate(env, contract, agents, params.nb_evaluation_episodes, params.nb_steps,
                            params.contracting, logger)


if __name__ == '__main__':
    args = sys.argv
    seed = 1589174148213878
    for run in range(10):
        for num_agents in [4]:
            for gamma in [0.95, 0.96, 0.97, 0.98, 0.99]:
                for machine_steps_inactive in [12]:
                    for contracting_steps in [5]:
                        train(machine_steps_inactive, contracting_steps, seed, num_agents, gamma)
