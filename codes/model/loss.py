import torch
import torch.nn as nn
from codes.model.baseline import ReactiveBaselineTorch as ReactiveBaseline


class AdvantageEstimator(object):
    def __init__(self, path_length, gamma=1, Lambda=0.05):
        self.use_cuda=torch.cuda.is_available()
        self.path_length = path_length
        self.gamma = gamma
        self.baseline_net = ReactiveBaseline(Lambda)

    def step(self, reward):
        assert torch.isfinite(reward).all().item()
        cum_discounted_reward = self.calc_cum_discounted_reward(reward)
        self.baseline_net.update(cum_discounted_reward.mean())
        baseline = self.baseline_net.get_baseline_value()
        normalized_reward = self.normalize_reward(cum_discounted_reward, baseline)
        return normalized_reward

    def calc_cum_discounted_reward(self, rewards):
        """
        calculates the cumulative discounted reward.
        :param rewards:
        :param T:
        :param gamma:
        :return:
        """
        running_add = torch.zeros(rewards.shape[0], device=rewards.device)  # [B]
        cum_disc_reward = torch.zeros(rewards.shape[0], self.path_length, device=rewards.device)  # [B, T]
        cum_disc_reward[:,
        self.path_length - 1] = rewards  # set the last time step to the reward received at the last state
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def normalize_reward(self, cum_discounted_reward, baseline):
        # multiply with rewards
        final_reward = cum_discounted_reward - baseline
        reward_mean = final_reward.mean()
        # Constant added for numerical stability
        reward_std = final_reward.std() + 1e-6
        final_reward = (final_reward - reward_mean) / reward_std
        return final_reward


class PolicyGradientLoss(nn.Module):
    def __init__(self, decaying_beta):
        super(PolicyGradientLoss, self).__init__()
        self.decaying_beta = decaying_beta

    def forward(self, losses, logits,  normalized_rewards, global_step):
        losses = torch.stack(losses, 1)
        logprobs = torch.mean(losses * normalized_rewards)
        reg_loss = self._entropy(logits)
        beta = self.decaying_beta.step(global_step)
        final_loss = logprobs - beta*reg_loss
        return final_loss, reg_loss

    def _entropy(self, logits):
        all_logits = torch.stack(logits, dim=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_policy = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))  # scalar
        return entropy_policy
