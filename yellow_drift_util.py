#!/usr/bin/python3
import time

# Old way to import gym, but now we use gymnasium
# import gym

import gymnasium as gym

#---------------------------
# Helper functions
#---------------------------

'''@brief Describes the environment actions, observation states, and reward range
'''
def describe_env(env: gym.Env):

    # Use .unwrapped to bypass wrappers like TimeLimit
    base_env = env.unwrapped

    num_actions = base_env.action_space.n
    obs = base_env.observation_space
    num_obs = base_env.observation_space.n

    # Old way to get reward range, but now we use gymnasium which doesn't have this attribute
    # reward_range = base_env.reward_range

    # Safely get reward_range, defaulting to (-inf, inf) if it doesn't exist
    reward_range = getattr(base_env, "reward_range", (-float('inf'), float('inf')))
    
    action_desc = { 
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger"
    }
    print("Observation space: ", obs)
    print("Observation space size: ", num_obs)
    print("Reward Range: ", reward_range)
    
    print("Number of actions: ", num_actions)
    print("Action description: ", action_desc)
    return num_obs, num_actions


'''@brief Get the string description of the action
'''
def get_action_description(action):
    action_desc = { 
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger"
    }
    return action_desc[action]

'''@brief print full description of current observation
'''
def describe_obs(obs):
    obs_desc = {
        0: "Red",
        1: "Green",
        2: "Yellow",
        3: "Blue",
        4: "In taxi"
    }
    obs_dict = breakdown_obs(obs)
    print("Passenger is at: {0}, wants to go to {1}. Taxi currently at ({2}, {3})".format(
        obs_desc[obs_dict["passenger_location"]], 
        obs_desc[obs_dict["destination"]], 
        obs_dict["taxi_row"], 
        obs_dict["taxi_col"]))

'''@brief Takes an observation for the 'taxi-v3' environment and returns details observation space description
    @details returns a dict with "destination", "passenger_location", "taxi_col", "taxi_row"
    @see: https://gymnasium.farama.org/environments/toy_text/taxi/
'''
def breakdown_obs(obs):
    # ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination = X
    # X % 4 --> destination
    destination = obs % 4
    # X -= remainder, X /= 4
    obs -= destination
    obs /= 4
    # X % 5 --> passenger_location
    passenger_location = obs % 5
    # X -= remainder, X /= 5
    obs -= passenger_location
    obs /= 5
    # X % 5 --> taxi_col
    taxi_col = obs % 5
    # X -= remainder, X /=5 
    obs -= taxi_col
    # X --> taxi_row
    taxi_row = obs
    observation_dict= {
        "destination": destination, 
        "passenger_location": passenger_location,
        "taxi_row": taxi_row, 
        "taxi_col": taxi_col
    }
    return observation_dict


'''@brief simulate the environment with the agents taught policy
'''
def simulate_episodes(env, agent, num_episodes=3):
    for _ in range(num_episodes):
        done = False
        state, _ = env.reset()
        describe_obs(state)
        env.render()
        while not done:
            # Random choice from behavior policy
            action = agent.select_action(state)
            # take a step
            env.render()
            time.sleep(0.1)
            next_state, _, done, _, _ = env.step(action)
            state = next_state
        time.sleep(1.0)

def main():
    # Note: Use v3 for the latest version
    env = gym.make('Taxi-v3')
    num_obs, num_actions = describe_env(env)


    # TODO: Train
    # agent = Agent(num_obs, num_actions)
    # agent.train(env, 5000)
    
    # TODO: Simulate
    # Note how here, we change the render mode for testing/simulation
    env2 = gym.make('Taxi-v3', render_mode="human")
    # simulate_episodes(env2, agent)

if __name__=="__main__":
    main()