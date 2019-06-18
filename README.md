# VPG-PyTorch
A minimalistic implementation of Vanilla Policy Gradient with PyTorch

This is a simple implementation of the Vanilla Policy Gradient (VPG) approach for tackling the reinforcement learning problem. Policy Gradient methods are part of a broader class of methods called policy-based methods. These methods are currently used in many state-of-the-art algorithms as an alternative to value-based methods such as Q-learning.

Policy-Based Methods provide some advantages from value-based methods such as:
- simplicity: they directly optimize for the optimal policy rather than passing through a value or Q function.
- stochasticity: sometimes the optimal policy is a stochastic one (eg. Rock-Paper-Scissors), in such cases value based methods won't work because they can only provide deterministic policies, but Policy-Based methods most likely will.
- continous action spaces: both Policy and Value Based methods work with continous state spaces, but only the first ones work with continous action spaces.

## Vanilla Policy Gradient
In the Vanilla Policy Gradient algorithm the agent interacts with the environment by sampling an action from a set of probabilies generated by the policy (probability distributions are generated in the case of continous action spaces).
It then stores state, action and reward at every step. At the end of every episode it updated the weights (θ) of the policy by computing the gradient of J(θ) (expected return function) and using gradient ascent to find the maximum of the function.

The pseudo-code for the algorithm is the following:

![alt text](https://spinningup.openai.com/en/latest/_images/math/47a7bd5139a29bc2d2dc85cef12bba4b07b1e831.svg "Policy Gradient Algorithm")

*Image taken from [OpenAI's Spinning Up](https://spinningup.openai.com)*

In this implementation of the algorithm, point 8 is skipped because we use the actual return of the episode (or the rewards to go) as our advantage function.

## Other Resources
For further study policy gradients, check out these links:
- https://www.youtube.com/watch?v=XGmd3wcyDg8&list=PLkFD6_40KJIxJMR-j5A1mkxK26gh_qg37&index=21
- https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
- https://www.youtube.com/watch?v=fdY7dt3ijgY
