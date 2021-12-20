import pickle
import random

import torch
from torch import autograd
from torch.nn.utils import parameters_to_vector

from src.constants import TASK_NAME, SEED, EPSILON, GAMMA, LOGS_FOLDER


def make_env(benchmark, test=False):
    if test:
        env = benchmark.test_classes[TASK_NAME]()
    else:
        env = benchmark.train_classes[TASK_NAME]()
    env.seed(SEED)
    if test:
        env.set_task(random.choice(benchmark.test_tasks))
    else:
        env.set_task(random.choice(benchmark.train_tasks))
    return env


def save_models(learner, critic, learner_label, critic_label):
    with open(f"{LOGS_FOLDER}{learner_label}", "wb") as f:
        pickle.dump(learner, f)
    with open(f"{LOGS_FOLDER}{critic_label}", "wb") as f:
        pickle.dump(critic, f)


def load_models(learner_label, critic_label):
    with open(f"{LOGS_FOLDER}{learner_label}", "rb") as f:
        learner = pickle.load(f)
    with open(f"{LOGS_FOLDER}{critic_label}", "rb") as f:
        critic = pickle.load(f)
    return learner, critic


def reshape(x):
    if len(x.size()) == 1:
        return x.view(-1, 1)
    return x


def normalize(x, epsilon=EPSILON):
    if x.numel() < 2:
        return x
    return (x - x.mean()) / (x.std() + epsilon)


def parameters_to_vector(parameters):
    parameters = [p.contiguous() for p in parameters]
    return torch.nn.utils.parameters_to_vector(parameters)


def hessian_vector_product(loss, parameters, other, damping=EPSILON):
    if not isinstance(parameters, torch.Tensor):
        parameters = list(parameters)

    grad_loss = autograd.grad(loss, parameters, create_graph=True)
    grad_loss = parameters_to_vector(grad_loss)

    grad_prod = torch.dot(grad_loss, other)
    hessian_prod = autograd.grad(grad_prod, parameters, retain_graph=True)
    hessian_prod = parameters_to_vector(hessian_prod)
    hessian_prod = hessian_prod + damping * other

    del grad_loss, grad_prod
    return hessian_prod


def conjugate_gradient(old_kl, learner, b, num_iterations=10, eps=EPSILON):
    x = torch.zeros_like(b)
    r = b
    p = r
    r_dot = torch.dot(r, r)
    for _ in range(num_iterations):
        Ap = hessian_vector_product(old_kl, learner.parameters(), p)
        alpha = r_dot / (torch.dot(p, Ap) + eps)
        x += alpha * p
        r -= alpha * Ap
        new_r_dot = torch.dot(r, r)
        p = r + (new_r_dot / r_dot) * p
        r_dot = new_r_dot
        if r_dot.item() < eps:
            break
    return x


def discount(rewards, dones):
    rewards = reshape(rewards)
    dones = reshape(dones)

    R = torch.zeros_like(rewards)
    discounted = torch.zeros_like(rewards)
    length = discounted.size(0)

    for t in reversed(range(length)):
        R = R * (1.0 - dones[t])
        R = rewards[t] + GAMMA * R
        discounted[t] += R[0]
    return discounted


def compute_advantages(critic, rewards, dones, states, next_states):
    returns = discount(rewards, dones)
    critic.update(states, returns)

    values = critic(states)
    next_values = critic(next_states)
    values = values * (1.0 - dones) + next_values * dones
    values = values.squeeze()
    next_value = torch.zeros(1, device=values.device)

    rewards = reshape(rewards)
    dones = reshape(dones)
    values = reshape(values)
    next_value = reshape(next_value)
    next_values = torch.cat((values[1:], next_value), dim=0)

    td_errors = rewards + GAMMA * (1.0 - dones) * next_values
    advantages = discount(td_errors - values, dones)
    return advantages


def update_module(module, memory=None):
    if memory is None:
        memory = {}

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p is not None and hasattr(p, 'update') and p.update is not None:
            if p in memory:
                module._parameters[param_key] = memory[p]
            else:
                updated = p + p.update
                memory[p] = updated
                module._parameters[param_key] = updated
                del p.update

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff is not None and hasattr(buff, 'update') and buff.update is not None:
            if buff in memory:
                module._buffers[buffer_key] = memory[buff]
            else:
                updated = buff + buff.update
                memory[buff] = updated
                module._buffers[buffer_key] = updated
                del buff.update

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            memory=memory,
        )

    del memory
    return module


# https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py
def clone_module(module, memo=None):
    if memo is None:
        memo = {}

    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )
    return clone
