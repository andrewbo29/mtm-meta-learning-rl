import gc
import random
from copy import deepcopy

import metaworld
import numpy as np
import pandas as pd
import torch
from torch import autograd
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

from src.actor import Actor
from src.constants import CUDA, EVAL_BATCH_SIZE, ADAPT_STEPS, ADAPT_BATCH_SIZE, TASK_NAME, SEED, NUM_ITERATIONS, \
    META_BATCH_SIZE, MAX_KL, LS_MAX_STEPS, LOGS_FOLDER, LOGS_ITERATION, LS_LR_ALPHA, META_LR
from src.critic import Critic
from src.maml import fast_adapt, maml_loss
from src.runner import Runner
from src.utils import make_env, hessian_vector_product, conjugate_gradient, save_models


# Meta-Testing
def evaluate(benchmark, learner, critic):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if CUDA else "cpu")
    env = make_env(benchmark, test=True)
    eval_task_list = random.choices(benchmark.test_tasks, k=EVAL_BATCH_SIZE)

    tasks_reward = 0.
    tasks_success = 0.

    for i, task in enumerate(eval_task_list):
        adapter = deepcopy(learner)
        for target_p, p in zip(adapter.parameters(), learner.parameters()):
            target_p.data.copy_(p)

        env.set_task(task)
        env.reset()
        task = Runner(env, device)

        for step in range(ADAPT_STEPS):
            gc.collect()
            torch.cuda.empty_cache()
            adapt_episodes = task.run(adapter, episodes=ADAPT_BATCH_SIZE)
            adapter = fast_adapt(adapter, adapt_episodes, critic)
            del adapt_episodes

        eval_episodes = task.run(adapter, episodes=ADAPT_BATCH_SIZE)
        del adapter
        del task

        task_success = torch.sum(eval_episodes.success).item() / ADAPT_BATCH_SIZE
        task_reward = torch.sum(eval_episodes.reward).item() / ADAPT_BATCH_SIZE
        del eval_episodes
        print(f"Task {i} - Reward: {task_reward:.6f}, Success : {task_success * 100:.2f}%")
        tasks_reward += task_reward
        tasks_success += task_success

    env.close()
    del env

    final_eval_reward = tasks_reward / EVAL_BATCH_SIZE
    final_eval_success = tasks_success / EVAL_BATCH_SIZE

    print(f"Mean Reward: {final_eval_reward:.6f}")
    print(f"Mean Success: {final_eval_success * 100:.2f}%")

    gc.collect()
    torch.cuda.empty_cache()

    return final_eval_reward, final_eval_success


def main():
    benchmark = metaworld.ML1(TASK_NAME)
    env = make_env(benchmark)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device("cuda" if CUDA else "cpu")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # learner, critic = load_models("best_eval_reward_learner.pkl", "best_eval_reward_critic.pkl")
    learner = Actor(state_size, action_size, device=device).to(device)
    critic = Critic(state_size, device=device)

    train_reward, train_success = [], []
    eval_reward, eval_success = [], []
    best_train_reward, best_train_success = 0, 0
    best_eval_reward, best_eval_success = 0, 0

    for iteration in range(NUM_ITERATIONS):
        gc.collect()
        torch.cuda.empty_cache()

        iteration_reward = 0.0
        iteration_success = 0.0
        iteration_replays = []
        iteration_policies = []

        # Meta-Training
        for task_config in tqdm(random.choices(benchmark.train_tasks, k=META_BATCH_SIZE), leave=False, desc='Data'):
            gc.collect()
            torch.cuda.empty_cache()

            adapter = deepcopy(learner)
            for target_p, p in zip(adapter.parameters(), learner.parameters()):
                target_p.data.copy_(p)

            env.set_task(task_config)
            env.reset()
            runner = Runner(env, device)

            replay = []

            # Inner Loop / Fast Adapt
            for step in range(ADAPT_STEPS):
                gc.collect()
                torch.cuda.empty_cache()

                train_episodes = runner.run(adapter, episodes=ADAPT_BATCH_SIZE)
                adapter = fast_adapt(adapter, train_episodes, critic)
                replay.append(train_episodes)
                del train_episodes

            valid_episodes = runner.run(adapter, episodes=ADAPT_BATCH_SIZE)
            del runner
            replay.append(valid_episodes)

            iteration_reward += torch.sum(valid_episodes.reward).item() / ADAPT_BATCH_SIZE
            iteration_success += torch.sum(valid_episodes.success).item() / ADAPT_BATCH_SIZE
            iteration_replays.append(replay)
            iteration_policies.append(adapter)

        print(f'\nIteration {iteration}')
        validation_reward = iteration_reward / META_BATCH_SIZE
        validation_success = iteration_success / META_BATCH_SIZE
        train_reward.append(validation_reward)
        train_success.append(validation_success)

        if validation_success > best_train_success:
            print("Saving best train success")
            save_models(learner, critic, "best_train_success_learner.pkl", "best_train_success_critic.pkl")
            best_train_success = validation_success

        if validation_reward > best_train_reward:
            print("Saving best train reward")
            save_models(learner, critic, "best_train_reward_learner.pkl", "best_train_reward_critic.pkl")
            best_train_reward = validation_reward

        print(f'Validation reward: {validation_reward:.6f}, success: {validation_success * 100:.2f}%')

        # Outer Loop / TRPO
        # Estimate policy gradient g_k
        # theta_old, D_KL old
        old_loss, old_kl = maml_loss(iteration_replays, iteration_policies, learner, critic)
        grad = autograd.grad(old_loss, learner.parameters(), retain_graph=True)
        grad = parameters_to_vector([g.detach() for g in grad])

        # Use CG to obtain x_k
        step = conjugate_gradient(old_kl, learner, grad)

        # Estimate proposed step delta_k
        # KL-divergence Hessian-vector product function f_v
        shs = 0.5 * torch.dot(step, hessian_vector_product(old_kl, learner.parameters(), step))
        lagrange_multiplier = torch.sqrt(shs / MAX_KL)
        step = step / lagrange_multiplier
        step_ = [torch.zeros_like(p.data) for p in learner.parameters()]
        vector_to_parameters(step, step_)

        # Compute proposed policy step delta_k
        step = step_
        del old_kl, grad
        old_loss.detach_()

        # Perform backtracking Line Search for TRPO to obtain final update
        for ls_step in range(LS_MAX_STEPS):
            gc.collect()
            torch.cuda.empty_cache()

            # Clone our policy to compute proposed update theta
            adapter = deepcopy(learner)
            for target_p, p in zip(adapter.parameters(), learner.parameters()):
                target_p.data.copy_(p)
            for param, delta in zip(adapter.parameters(), step):
                param.data.add_(other=delta.data, alpha=-META_LR * LS_LR_ALPHA ** ls_step)

            # Compute proposed update theta, new_loss
            new_loss, kl = maml_loss(iteration_replays, iteration_policies, adapter, critic)
            del adapter

            # Verification
            if new_loss < old_loss and kl < MAX_KL:
                for param, delta in zip(learner.parameters(), step):
                    param.data.add_(other=delta.data, alpha=-META_LR * LS_LR_ALPHA ** ls_step)
                break

            del new_loss, kl

        del old_loss
        del iteration_replays, iteration_policies

        gc.collect()
        torch.cuda.empty_cache()

        if iteration % LOGS_ITERATION == 0:
            r, s = evaluate(benchmark, learner, critic)
            eval_reward.append(r)
            eval_success.append(s)

            train_df = pd.DataFrame({
                'reward': train_reward,
                'success': train_success
            })
            train_df.to_csv(f"{LOGS_FOLDER}train_log.csv", index=False)

            eval_df = pd.DataFrame({
                'reward': eval_reward,
                'success': eval_success
            })
            eval_df.to_csv(f"{LOGS_FOLDER}eval_log.csv", index=False)

            if s > best_eval_success:
                print("Saving best eval success")
                save_models(learner, critic, "best_eval_success_learner.pkl", "best_eval_success_critic.pkl")
                best_eval_success = s

            if r > best_eval_reward:
                print("Saving best eval reward")
                save_models(learner, critic, "best_eval_reward_learner.pkl", "best_eval_reward_critic.pkl")
                best_eval_reward = r

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
