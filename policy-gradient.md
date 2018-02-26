# Reinforcement Learning: Policy Gradient (Lecture 7)

Policy Gradient methods attempt to perform *gradient ascent* on the policy
directly (i.e. improving it directly). The basic idea is to parameterize the
policy, i.e. to have $\pi_{\theta}(a | s)$.

The advantage of this is, for one, that you need no longer store the value
function, but only the policy function. More formally, policy gradient methods
have the benefit of *better convergence properties*, higher effectiveness in
*high-dimensional or continuous* action spaces and the ability to learn
*stochastic policies*. There are also disadvantages, however, such as that
policy gradient methods may converge to a local minimum rather than a global one
and the fact that evaluating a policy may be inefficient and high variance.

One of the most important properties is the better performance in high
dimensional spaces. The reason why is that for value-based methods we always
updated our value-function with some maximization over Q-values:

$$Q(s, a) \gets Q(s, a) + \alpha(R_s^a + \gamma\max_{a'}Q(s', a') - Q(s, a)).$$

When this maximization is expensive, policy-gradient methods are favored. Also,
the notion of a stochastic policy $\pi(a | s)$ as the final, output, policy,
instead of a deterministic, greedy policy can be more suitable in *partially
observable Markov decision processes* (*POMDPs*). In such POMDP environments,
aggressive ("stubborn") greedy policies may lead to very bad decisions or
infinite loops, simply because we are trying to be certain about what actions to
make in an environment where we do not actually have all the required
information.

Policy-based learning can be slower, but more exact, because it is less
aggressive.

## Policy Objective Functions

Once more, the goal in policy-gradient methods is to nudge the policy towards a
*higher* value, i.e. to maximize it. For this, we first need to define some
optimization goal $J(\theta)$ where $\theta$ are also the weights that
parameterize the policy $\pi_{\theta}$. Examples of this are:

- Optimizing the expected return of the *first states*:
  $$J_1(\theta) = V^{\pi_{\theta}}(s_1) = \mathbb{E}_{\pi_{\theta}}(v_1)$$
- Optimizing the expected average *return*:
  $$J_{\text{av}V}(\theta) = \sum_s d^{\pi_{\theta}}(s) V^{\pi_{\theta}}(s)$
- Optimizing the expected average *reward*:
  $$J_{\text{av}R}(\theta) = \sum_s d^{\pi_{\theta}}(s)\sum_a \pi_{\theta}(s, a)\mathcal{R}_s^a$$

where $d^{\pi_{\theta}}$ is a distribution over states. Either way, we will want
to optimize one of these objectives and find the $\theta$ that maximizes
$J(\theta)$. Approaches for this include the non-gradient based methods like the
simplex method and or genetic algorithms, or gradient-based methods like
gradient descent, conjugate gradient or Quasi-Newton methods. We focus mostly on
gradient descent.

### Finite Differences

The first method of finding the gradient $\nabla_{\theta} J(\theta) $to do
gradient descent that we want to investigate is the simple *finite differences*
method. In this method, to find the gradient, we estimate each partial
derivative separately using the rule

$$\frac{\partial J(\theta)}{\partial \theta_k} \approx \frac{J(\theta + \varepsilon u_k) - J(\theta)}{\varepsilon},$$

basically taking a small difference quotient. Here $u_k = (0, ..., 1, ...,
0)^\top$ is a "one-hot-encoded" vector that adds the change $\varepsilon$ just
to the $k$-th component. For $n$ dimensions, this has to be done $n$ times. As
such, if $J(\theta)$ is expensive to evaluate, this method is not efficient.
Nevertheless, it can be a good choice because it is simple and sometimes just
works.

### Analytic Solutions

While the finite differences method may work in simple cases, it is not
generally feasible. For this, we'll want to look at analytic solutions and
assume that our objective function is differentiable and we know how to compute
the gradient. Before that, however, it is better to express our objective
function as a *score function*. For this, we perform the following
transformation:

$$
\begin{align}
  \nabla_{\theta}\pi_{\theta}(s, a) &= \pi_{\theta}(s, a)\frac{\nabla_{\theta}\pi_{\theta}(s, a)}{\pi_{\theta}} \\
  &= \pi_{\theta}(s, a)\nabla_{\theta}\log \pi_{\theta}(s, a)
\end{align}
$$

which holds since $\nabla \log f(x) = \frac{1}{f(x)}f'(x)$ by the chain rule.
This new objective $\pi_{\theta}(s, a)\nabla_{\theta}\log \pi_{\theta}(s, a)$ is
now called the score function. The __whole point__ of this is to have the
probability $\pi_{\theta}(s, a)$ in the term again (as opposed to just $\nabla
\pi_{\theta}(s, a)$), so that we can take expectations and not have to deal with
gradients directly.

#### Softmax Policy

A softmax policy always involves some kind of exponentiation. For example, we
could weight the features of our actions linearly. Let $\varphi(s,
a)^\top\theta$ be such a linear weighting, where $\varphi(s, a)$ is a feature
vector dependent on a $(\text{state, action})$ pair. The softmax policy would
then look something like this:

$$
\pi_{\theta}(s, a) \propto e^{\varphi(s, a)^\top \theta}
$$

where now the probability of an action is equal to its exponentiated weight.
Then the score function is

$$
\nabla_{\theta}\log \pi_{\theta}(s, a) = \varphi(s, a) - \mathbb{E}_{\pi_{\theta}}[\varphi(s, \cdot)]
$$

which is basically asking "how different is the action $a$ from all other
actions?".

#### Gaussian Policy

The second example of a policy is a *Gaussian policy*, which distributes actions
according to a normal distribution. For this, we now look at features of a
state: $\varphi(s)$. Then, we keep a separate Gaussian for each state by setting
the mean of each Gaussian to be some linear combination of the features of the
state: $\mu(s) = \varphi(s)^\top\theta$. The variance $\sigma^2$ is fixed or,
sometimes, also parameterized. As such, the occurrence of an action in a
particular state would be distributed according to a Gaussian:

$$a \sim \mathcal{N}(\mu(s), \sigma^2).$$

The score function is then

$$\nabla_{\theta}\log \pi_{\theta}(s, a) = \frac{(a - \mu(s))\varphi(s)}{\sigma^2}$$

where $a$ would be such a continuous action sampled from $\mathcal{N}$. Again,
we are measuring "how far apart" a particular action is from the mean.

#### Computing the Policy Gradient

Consider a simple MDP with just one state and some actions. Then we can define
the policy objective function for the average reward as

$$J(\theta) = \mathbb{E}_{\pi_{\theta}}[r] = \sum_{s \in \mathcal{S}}d(s)\sum_{a \in \mathcal{A}}\pi_{\theta}(s, a)\mathcal{R}_{s, a}$$

and the gradient subsequently as

$$
\begin{align}
\nabla_{\theta}J(\theta) &= \sum_{s \in \mathcal{S}}d(s)\sum_{a \in \mathcal{A}}\nabla_{\theta}\pi_{\theta}(s, a)\mathcal{R}_{s, a} \\
&= \sum_{s \in \mathcal{S}}d(s)\sum_{a \in \mathcal{A}}\pi_{\theta}(s, a)\nabla_{\theta}\log\pi_{\theta}(s, a)\mathcal{R}_{s, a} \\
&= \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s, a) \cdot r]
\end{align}
$$

where we see the real point of this likelihood ratio transformation is that we
can define an expectation like this. The next step then is to not only define an
expectation over the next step, but over any number of steps. For this, we
basically just replace the one-step reward $r$ with a Q-value
$Q^{\pi_{\theta}}(s, a)$:

$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s, a) \cdot Q^{\pi_{\theta}}(s, a)]
$$

#### Reinforce

At this point, we can define the simplest of policy gradient algorithms known as
*REINFORCE*. This algorithm is a Monte-Carlo approach that samples an episode
and computes the total return $G_t$ for that episode. It then simply adjusts the
weights in the direction of the gradient of the policy as we outlined above:

$$\theta \gets \theta + \alpha\nabla_{\theta}\log \pi_{\theta}(s_t, a_t) G_t$$

note that $G_t$ is the total accumulated reward *starting* at step $t$, up to
the terminal state $T$ (i.e. it is not the same for each state).

or, in an algorithm:

1. Initialize $\theta$ arbitrarily,
2. Sample an episode $\{s_1,a_1,r_2,...,s_{T-1},a_{T-1},r_T\} \sim \pi_{\theta}$,
3. For each transition $t \in [1, T - 1]$:
    1. Perform the REINFORCE update:
    $$\theta \gets \theta + \alpha\nabla_{\theta}\log \pi_{\theta}(s_t, a_t) G_t$$

here the total return $G_t$ plays the role of changing the sign of the gradient
and impacting the magnitude. Recall again that $G_t$ starts at $t$ and goes to
$T$ and is different for each transition in the episode.

### Actor Critic

The problem with the Monte-Carlo based REINFORCE algorithm outlined above is
that it is very slow (because of full rollouts) and high variance. A solution to
this is the *Actor-Critic* family of algorithms which essentially combines the
concepts of policy gradient and policy iteration with value approximation.

Basically, instead of calculating $Q^{\pi_{\theta}}(s, a)$ or (in the limit of
that) $G_t$, we will now have a (linear/NN) approximation $Q_w(s, a)$ of the
value function. This approximation is called the *critic* and conceptually
evaluates the actions of the *actor*, which is the policy by which we act. As
such, we will now have two sets of parameters $w$ and $\theta$. $w$ will be the
parameters or weights of the approximation $Q_w(s, a)$, while $\theta$ will be
the parameters of our differentiable policy $\pi$. The idea then, is that the
critic will supply the Q-values for our updates:

$$
\begin{align}
\nabla_{\theta}J(\theta) &\approx \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log\pi_{\theta}(s, a)Q_w(s, a)] \\
\Delta\theta &= \alpha\nabla_{\theta}\log\pi_{\theta}(s, a) Q_w(s, a)
\end{align}
$$

The way we find $Q_w(s, a)$ is now just the same problem we had in the
discussion of value approximation. We could use a linear combination or neural
network for the approximation and then train it with Monte-Carlo or Temporal
Difference learning methods. For example, if we use TD(0), we'll start in some
state $s$ and take action $a$ to end up in state $s'$. We'll then sample a new
action $s'$ from $\pi_{\theta}$ (!) and correct our value estimate $Q_w(s, a)$.
At the same time, we use the (old) value of $Q_w(s, a)$ to correct our policy.
Note that we do not use a greedy policy to correct $Q_w$ like in Q-learning,
because we now have an __explicit__ policy $\pi$ that we can follow. Note also
how $Q_w$ and $\pi_{\theta}$ are actually learned both, simultaneously.

The above description of the TD(0) variant of this algorithm maps to the
following algorithm:

1. Initialize the state $s$ and parameters $\theta$
2. Sample $a \sim \pi_{\theta}$
3. For each time step $t$:
    1. Sample reward $r = R_s^a$ and transition state $s' \sim P_s^a$
    2. Sample action $a' \sim \pi_{\theta}(s', a')$
    3. $\delta = r + \gamma Q_w(s', a') - Q_w(s, a)$ # TD target
    4. $\theta = \theta + \alpha\nabla_{\theta}\log\pi_{\theta}(s, a)Q_w(s, a)$
    5. $w \gets w + \beta\delta\varphi(s, a)$
    6. $a \gets a', s \gets s'$

#### Reducing Variance

One problem of the Actor Critic Model may be variance. A way to mitigate this
problem is to *subtract a baseline*. Basically, what is possible is instead of
defining our objective function in terms of $Q^{\pi_{\theta}}$ directly, we can
define some *baseline function* $B(s)$ that is a function only of the state and
then optimize w.r.t. $Q^{\pi_{\theta}} - B(s)$. The reason for this is that if
this baseline changes only as a function of the state, it will not influence the
gradient or optimization process in any way (because it does not depend on
$\theta$), but still reduces variance.

A good baseline function is the state-value function $V^{\pi_{\theta}}(s)$. As
such, if we subtract this baseline function from our action-value function, we
get

$$A^{\pi_{\theta}}(s, a) = Q^{\pi_{\theta}}(s, a) - V^{\pi_{\theta}}(s).$$

This function is called the *advantage function*. It basically gives an
indication of how much better it is to take action $a$ as opposed to just
being/staying in state $s$. As such, we can now rewrite our optimization
objective as

$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}\log\pi_{\theta}(s, a)A^{\pi_{\theta}}(s, a)]
$$

Of course, we must now ask how we get $V^{\pi{\theta}}(s, a)$ to compute the
advantage function. One way is to also learn $V$ with another approximator and
another set of weights. However, if we think back to the definition of the
action-value function as the reward of an action plus the maximum/average
state-value of all subsequent states, we also remember that we actually only
need the value function to compute Q-values. As such, and if we are willing to
use the advantage function $A$, we can define our TD error as

$$\delta^{\pi_{\theta}} = r + \gamma V^{\pi_{\theta}}(s') - V^{\pi_{\theta}}(s)$$

since

$$
\begin{align}
  \mathbb{E}[\delta^{\pi_{\theta}} | s,a ] &= \mathbb{E}_{\pi_{\theta}}[r + \gamma V^{\pi_{\theta}}(s') | s,a] - V^{\pi_{\theta}}(s) \\
  &= Q^{\pi_{\theta}}(s, a) - V^{\pi_{\theta}}(s) \\
  &= A^{\pi_{\theta}}(s, a).
\end{align}
$$

The policy gradient then becomes

$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{\theta_{\pi}}[\nabla_{\theta}\log \pi_{\theta}(s, a)\delta^{\pi_{\theta}}].
$$

As a result, we can now have just one set of weights $v$ and approximate the value function

$$V_v(s) \approx V_{\pi}(s)$$
