import numpy as np
import json
from copy import deepcopy


class Contract:

    def __init__(self,
                 policy_nets,
                 contracting_net,
                 agents,
                 gamma,
                 contracting_target_update=0,
                 nb_contracting_steps=10,
                 mark_up=1.0,
                 contracting=0,
                 render=True):

        with open('envs/actions.json', 'r') as f:
            actions_json = json.load(f)

        self.policy_nets = policy_nets
        self.contracting_net = contracting_net
        self.gamma = gamma
        self.nb_agents = agents
        self.contracting_target_update = contracting_target_update
        self.nb_contracting_steps = nb_contracting_steps
        self.mark_up = mark_up
        self.render = render
        self.num_contracts = 0
        self.contracting = contracting
        self.contracting_pairs = []
        self.current_contracts = np.zeros(self.nb_agents, dtype=int)
        self.greedy = np.zeros(self.nb_agents, dtype=int)
        self.pos_rewards = np.zeros((self.nb_agents, self.nb_contracting_steps))
        self.neg_rewards = np.zeros((self.nb_agents, self.nb_contracting_steps))
        self.compensations = np.zeros((self.nb_agents, self.nb_contracting_steps))
        if self.contracting == 0:
            self.actions = actions_json['no_contracting_actions']
        if self.contracting == 1:
            self.actions = actions_json['contracting_actions']
        self.done = np.zeros(self.nb_agents)

    def contracting_n_steps(self, env, observations, actions, contracting_mode):

        num_contracts, contracting_indices, offerings, acceptances = self.check_contracting(actions, observations, contracting_mode)

        if np.sum(self.greedy) == 0:
            observations, reward, done, info = env.step(actions)
            info['contracting'] = 0
            info['transfer'] = 0
            info['offerings'] = offerings
            info['acceptances'] = acceptances
            return observations, reward, done, info, []

        elif np.sum(self.greedy) > 0:

            c_actions = []
            current_observation = deepcopy(observations)
            for i in range(self.nb_agents):
                if self.greedy[i] and i in np.array(self.contracting_pairs).flatten():
                    c_actions.append(self.policy_nets[i].policy(observations[i]))
                    self.current_contracts[i] -= 1
                elif i in np.array(self.contracting_pairs).flatten():
                    self.current_contracts[i] -= 1
                    q_vals = self.policy_nets[i].compute_q_values(observations[i])
                    c_actions.append(np.argmin(q_vals))
                    c_t = np.maximum(np.max(q_vals) - np.min(q_vals), 0)
                    self.compensations[i][self.current_contracts[i]] = c_t
                else:
                    c_actions.append(actions[i])

            observations, reward, done, info = env.step(c_actions)

            info['offerings'] = 0
            info['acceptances'] = 0

            info['offerings'] += offerings
            info['acceptances'] += acceptances

            # Save reward of contract partners until contract terminates
            for i in np.array(self.contracting_pairs).flatten():
                if reward[i] >= 0:
                    self.pos_rewards[i][self.current_contracts[i]] += reward[i]
                else:
                    self.neg_rewards[i][self.current_contracts[i]] += reward[i]
                reward[i] = 0

            self.compensations[i] *= 1 / self.gamma

            info['transfer'] = 0

            pairs = deepcopy(self.contracting_pairs)
            for pair in pairs:
                if (self.current_contracts[pair[0]] == 0) or \
                        any(done is True for done in [done[pair[0]], done[pair[1]]]):
                    reward, transfer = self.compute_rewards(reward, pair)
                    self.reset_contract_status(pair)
                    self.contracting_pairs.remove(pair)
                    info['transfer'] += transfer
            info['contracting'] = num_contracts
            self.num_contracts += num_contracts
            self.done = done
            if not all(done is True for done in self.done):
                self.contracting_net.save(current_observation,
                                          contracting_indices,
                                          observations,
                                          reward,
                                          done,
                                          self.num_contracts)
                self.contracting_net.train()

            return observations, reward, done, info, np.array(self.contracting_pairs).flatten()

    def reset_done(self):
        self.done = np.zeros(self.nb_agents)

    # Accumulates reward and compensation of contract period. Called, when contract terminates
    def compute_rewards(self, reward, pair):
        accumulated_reward = np.sum(self.compensations[pair[1]])
        transfer = np.minimum(np.sum(self.pos_rewards[pair[0]]), accumulated_reward)
        reward[pair[0]] = np.sum(self.pos_rewards[pair[0]]) + np.sum(self.neg_rewards[pair[0]]) - transfer
        reward[pair[1]] = np.sum(self.pos_rewards[pair[1]]) + np.sum(self.neg_rewards[pair[1]]) + transfer
        return reward, transfer

    # Resets contract status and contracting variables. Called, when contract terminates
    def reset_contract_status(self, pair):
        self.current_contracts[pair[0]] = 0
        self.current_contracts[pair[1]] = 0
        self.greedy[pair[0]] = 0
        self.greedy[pair[1]] = 0
        self.pos_rewards[pair[0]] = 0
        self.pos_rewards[pair[1]] = 0
        self.neg_rewards[pair[0]] = 0
        self.neg_rewards[pair[1]] = 0
        self.compensations[pair[0]] = 0
        self.compensations[pair[1]] = 0
        self.num_contracts -= 1

    # Checks and establishes contracts, if there is supply and demand
    def check_contracting(self, actions, observations, contracting_mode):
        num_contracts = 0
        contracting_indices = 0
        offerings = 0
        acceptances = 0

        if contracting_mode == 0:
            pass

        if contracting_mode == 1:
            contracting_indices = self.contracting_net.policy(observations)
            for agent in range(self.nb_agents):
                contracting_partner = contracting_indices[agent]
                permission, offerings, acceptances = self.check_contracting_permission(agent, contracting_partner, actions)
                if permission:
                    self.greedy[agent] = 1
                    self.contracting_pairs.append([agent, contracting_partner])
                    self.current_contracts[agent] = self.nb_contracting_steps
                    self.current_contracts[contracting_partner] = self.nb_contracting_steps
                    num_contracts += 1
            assert any(i == 0 for i in self.greedy)
        return num_contracts, contracting_indices, offerings, acceptances

    # Check if potential contracting pairs are even possible. Called in check_contracting
    def check_contracting_permission(self, agent, contracting_partner, actions):
        offerings = 0
        acceptances = 0
        if self.actions[actions[agent]][3] == 1:
            offerings += 1
        elif self.actions[actions[agent]][2] == 1:
            acceptances += 1

        different_agents = (agent != contracting_partner)
        supply_and_demand = self.actions[actions[agent]][3] == 1 and self.actions[actions[contracting_partner]][2] == 1
        currently_contracting = any(x in np.array(self.contracting_pairs).flatten()
                                    for x in [agent, contracting_partner])
        agent_done = any(done is True for done in [self.done[agent], self.done[contracting_partner]])
        if supply_and_demand and different_agents and not currently_contracting and not agent_done:
            return True, offerings, acceptances
        else:
            return False, offerings, acceptances


def make_contract(params, policy_nets, contracting_net):
    contract = Contract(policy_nets=policy_nets,
                        contracting_net=contracting_net,
                        agents=params.nb_agents,
                        gamma=params.gamma,
                        contracting_target_update=params.contracting_target_update,
                        nb_contracting_steps=params.nb_contracting_steps,
                        mark_up=params.mark_up,
                        contracting=params.contracting,
                        render=False)

    return contract
