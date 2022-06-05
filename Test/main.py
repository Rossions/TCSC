import numpy as np
import torch
import matplotlib.pyplot as plt
import traci
import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from env import environment
from agent.TD3_twins_action import TD3_twins_action
from agent import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    env_with_dead = False
    cfg = 'C:/Users/14777/codeStation/TCSC/Test/data/cfg.sumocfg'
    env = environment.SumoEnvironment(cfg)
    state_dim = env.observation_space
    action_dim = env.action_space
    phase_action_dim = env.phase_action_space
    duration_action_dim = env.duration_action_space
    action_dim_high = env.action_space_range[0]
    action_dim_low = env.action_space_range[1]
    max_duration_action = float(action_dim_high)
    min_duration_action = float(action_dim_low)
    tls_id = traci.trafficlight.getIDList()[0]
    expl_noise = 0.25
    print('  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_duration_action, '  min_a:',
          min_duration_action)

    render = False
    Loadmodel = False
    ModelIdex = 3600  # which model to load
    # random_seed = seed

    Max_episode = 2000000
    save_interval = 400  # interval to save model

    # if random_seed:
    #     print("Random Seed: {}".format(random_seed))
    #     torch.manual_seed(random_seed)
    #     env.seed(random_seed)
    #     np.random.seed(random_seed)

    writer = SummaryWriter(log_dir='runs/exp')

    kwargs = {
        "env_with_dead": env_with_dead,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "phase_action_dim": phase_action_dim,
        "duration_action_dim": duration_action_dim,
        "max_action": max_duration_action,
        "min_action": min_duration_action,
        "gamma": 0.99,
        "a_lr": 1e-4,
        "c_lr": 1e-4,
        "Q_batchsize": 256,
    }
    model = TD3_twins_action(**kwargs)
    if Loadmodel: model.load(ModelIdex)
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, phase_action_dim, duration_action_dim, max_size=int(1e6))

    all_ep_r = []

    for episode in range(Max_episode):
        s, done = env.reset(tls_id), False
        ep_r = 0
        steps = 0
        expl_noise *= 0.999

        '''Interact & trian'''
        while not done:
            steps += 1
            # if render:
            #     a = model.select_action(s)
            #     s_prime, r, done, info = env.step(a)
            # else:
            p_a, d_a = model.select_action(s)
            d_a = (d_a + np.random.normal(0, max_duration_action * expl_noise, size=duration_action_dim)
                   ).clip(min_duration_action, max_duration_action)
            _, p_a = torch.topk(torch.from_numpy(p_a), 1)
            s_prime, r, done, info = env.step(tls_id, p_a, d_a)

            # Tricks for BipedalWalker
            if r <= -100:
                replay_buffer.add(s, p_a, d_a, r, s_prime, True)
            else:
                replay_buffer.add(s, p_a, d_a, r, s_prime, False)

            if replay_buffer.size > 2000: model.train(replay_buffer)

            s = s_prime
            ep_r += r

        '''plot & save'''
        if (episode + 1) % save_interval == 0:
            model.save(episode + 1)
            plt.plot(all_ep_r)
            plt.savefig('ep{}.png'.format(episode + 1))
            plt.clf()

        '''record & log'''
        # all_ep_r.append(ep_r)
        if episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        writer.add_scalar('s_ep_r', all_ep_r[-1], global_step=episode)
        writer.add_scalar('ep_r', ep_r, global_step=episode)
        writer.add_scalar('exploare', expl_noise, global_step=episode)
        print('episode:', episode, 'score:', ep_r, 'step:', steps, 'max:', max(all_ep_r))

    env.close()


if __name__ == '__main__':
    main()
