def play_random_till_episode_end(env, max_steps=10000000000):
    idx = 0
    while True:
        action = env.action_space.sample()
        ob, reward, done, _ = env._step(action)
        idx += 1
        if done or idx > max_steps:
            break
