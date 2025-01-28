from datetime import datetime
import numpy as np


def evaluate(env, contract, agents, nb_episodes, nb_steps, contracting_mode, logger):
    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    print("evaluation started at {}".format(exp_time))

    for evaluation_episode in range(nb_episodes):
        steps = 0
        states = env.reset()
        all_done = False
        episode_rewards = np.zeros(env.nb_agents, dtype=np.float32)
        episode_contracts = 0
        accumulated_transfer = 0
        offerings = 0
        acceptances = 0

        # classic reinforcement learning loop
        while not all_done and steps < nb_steps:
            steps += 1
            actions = []

            for i in range(env.nb_agents):
                actions.append(agents[i].policy(states[i]))

            next_state, reward, done, info, contracting_pairs = contract.contracting_n_steps(env, states, actions, contracting_mode)

            for agent_index in range(env.nb_agents):
                agents[agent_index].save(states[agent_index],
                                         actions[agent_index],
                                         next_state[agent_index],
                                         reward[agent_index],
                                         done[agent_index])
                episode_rewards[agent_index] += reward[agent_index]

            episode_contracts += info['contracting']
            accumulated_transfer += info['transfer']
            acceptances += info['acceptances']
            offerings += info['offerings']
            states = next_state
            all_done = all(done is True for done in done)

        contract.reset_done()


        if logger is not None:
            logger["eval/return"].append(np.sum(episode_rewards))
            logger["eval/steps"].append(steps)
            logger["eval/num_contracts"].append(episode_contracts)
            logger["eval/accumulated_transfer"].append(accumulated_transfer)
            logger["eval/num_contract_offer"].append(offerings)
            logger["eval/num_contract_accepts"].append(acceptances)

            for i in range(env.nb_agents):
                logger["eval/episode_return_agent-{}".format(i)].append(episode_rewards[i])

        print("evaluation {} finished at step {} with reward: {} at timestamp {}".format(
            evaluation_episode, steps, episode_rewards, exp_time))
