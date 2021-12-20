from torch import autograd
from torch.distributions.kl import kl_divergence
from tqdm import tqdm

from src.constants import ADAPT_LR
from src.utils import compute_advantages, normalize, update_module, clone_module


def a2c_loss(train_episodes, actor, critic):
    states = train_episodes.state
    actions = train_episodes.action
    rewards = train_episodes.reward
    dones = train_episodes.done
    next_states = train_episodes.next_state
    log_probs = actor.compute_proba(states, actions)

    advantages = compute_advantages(critic, rewards, dones, states, next_states)
    advantages = normalize(advantages).detach()
    loss = actor.loss(log_probs, advantages)

    return loss


def fast_adapt(actor, train_episodes, critic):
    loss = a2c_loss(train_episodes, actor, critic)
    gradients = autograd.grad(loss,
                              actor.parameters(),
                              retain_graph=True,
                              create_graph=True)

    if gradients is not None:
        for param, grad in zip(list(actor.parameters()), gradients):
            if grad is not None:
                param.update = -ADAPT_LR * grad
    updated_actor = update_module(actor)

    return updated_actor


def maml_loss(iteration_replays, iteration_policies, actor, critic):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in tqdm(zip(iteration_replays, iteration_policies),
                                         total=len(iteration_replays),
                                         desc='MAML Loss',
                                         leave=False):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]

        new_policy = clone_module(actor)
        for train_episodes in train_replays:
            new_policy = fast_adapt(new_policy, train_episodes, critic)

        states = valid_episodes.state
        actions = valid_episodes.action
        next_states = valid_episodes.next_state
        rewards = valid_episodes.reward
        dones = valid_episodes.done

        old_densities = old_policy.distribution(states)
        new_densities = new_policy.distribution(states)
        del new_policy
        kl = kl_divergence(new_densities, old_densities).mean()
        mean_kl += kl

        advantages = compute_advantages(critic, rewards, dones, states, next_states)
        advantages = normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
        del old_densities, new_densities
        mean_loss += actor.loss(old_log_probs, advantages, new_log_probs)
        del old_log_probs, new_log_probs, advantages

    mean_kl /= len(iteration_replays)
    mean_loss /= len(iteration_replays)
    return mean_loss, mean_kl
