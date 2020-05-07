import gym

env = gym.make('CartPole-v0')

# -------- State
print("\n State")
# get the starting state
# State is [cp, cv, pa, pv]
# with cp = Cart Position, cv = Cart Velocity, pa = Pole Angle, pv = Pole Velocity
start_state = env.reset()
print(start_state)

# Get information about the state vector
box = env.observation_space
print("Maximum Values: {}".format(box.high))
print("Minimum Values: {}".format(box.low))
print("Random State: {}".format(box.sample()))


# -------- Action
print("\n Action")
action_space = env.action_space
# contains 0 for pushing to the left, and 1 for push to the right
# so 2 actions
print("Number of possible actions: {}".format(action_space.n))


# -------- Take a single
print("\n Taking a single action")
# 0 is left, 1 is right
action = action_space.sample()
observation, reward, done, info = env.step(action)
print("Taking random action: {}".format(action))
print("Next State: {}".format(observation))
print("Reward: {}".format(reward))
print("Is done? : {}".format(done))
print("Information: {}".format(info))


# --------- Play a whole episode (until pole falls over) using random action
print("\n Playing an episode")
done = False
n = 0
while not done:
    n += 1
    action = action_space.sample()
    observation, reward, done, _ = env.step(action)
    print("Step: {}, Action: {}, Next State: {}, Reward: {}, Is done? {}".format(n, action, observation, reward, done))


# ------- Add a linear model to the action selection process
# we will select n random weight vectors, play the episode for each one and choose the best (sort of..)
