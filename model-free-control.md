# Reinforcement Learning: Model Free Control (Lecture 5)

$\newcommand{\argmax}{\mathop{\rm argmax}\nolimits}$

## On Policy Learning

### Monte Carlo Control

The first topic we want to explore for model free control is policy iteration
with Monte Carlo (MC) methods. The idea is that if we can do model-based policy
iteration (remember that a model means the transition probability matrix and
reward distribution are known), we should also be able to do model-free policy
iteration with Monte Carlo. If we just think of the simplest integration of MC,
we could rollout a full episode, receive a reward, backup (backpropagate) the
resulting state-values and then pick the action of the policy for the current
state greedily:

$$\pi'(s) = \argmax_{a \in \mathcal{A}} \mathcal{R}_{t+1}^a + \mathcal{P}_{ss'}^a V(S_{t+1})$$

The problem with this is that this update requires $R$ and $\mathcal{P}$, which
are both only available in a model-based system! However, we can use a trick. We
don't know $\mathcal{R}$ or $\mathcal{P}$, but we do know that the action-values
given by $Q(s, a)$ are defined in terms of these two quantities, as

$$Q(s, a) = \mathcal{R}_t^a + \gamma\sum_{s' \in \mathcal{S}}\mathcal{P}_{ss'}^a
v_\pi(s', a)$$

As such, we can hide the fact that we don't know the parameters of our model by
iteratively approximating the value of $Q(S_t, A_t)$ according to our policy
$\pi$, such that $Q = q_\pi$:

$$\pi'(s) = \argmax_{a \in \mathcal{A}} Q(s, a)$$

The Q-function itself we would compute iteratively in just the same way we
computed the value function iteratively for prediction. That is, we will now
update according to

$$Q(S_t, A_t) \gets S(S_t, A_t) + \frac{1}{N(S_t)}(G_t - Q(S_t, A_t)).$$

Again, we see that state-value functions ($v) and action-value functions ($q$)
are related in that they both approximate the total return.

It is important to understand that the major difference between this and policy
iteration is that we now no longer have full knowledge of the model. What this
means is that when finding the optimal policy, we cannot simply use a 100%
greedy strategy all the time, since all our action-values are *estimates*, while
they were *computed* earlier. As such, we now need to introduce an element of
exploration into our algorithm. For this, we choose what is called an
$\varepsilon$-greedy strategy. Such a strategy picks a random action with
probability $\varepsilon$ and the greedy action with probability
$1-\varepsilon$. Formally, for $m$ actions, this means

$$
\pi(a|s) =
\begin{cases}
  \varepsilon \cdot \frac{1}{m} + (1 - \varepsilon) \text{ if } a = \argmax_{a' \in \mathcal{A}} Q(s, a'), \\
  \varepsilon \cdot \frac{1}{m} \text{ otherwise.}
\end{cases}
$$

Note how any action has a probability of $\varepsilon \cdot \frac{1}{m}$ of
being picked in the random case. However the greedy option has an additional
chance of $1 - \varepsilon$ of being chosen.

There is, however, a further modification to our strategy we must make for our
policy and action-value function to converge. This modification must meet the
*GLIE* constraint, which stands for *Greedy in the Limit with Infinite
Exploration*. This constraint can be broken down into two further constraints:

1. *Infinite Exploration* means all $\text{state, action}$ pairs should be explored infinitely many times in the limit:
  $$\lim_{k\rightarrow\infty} N_k(s, a) = \infty$$
2. *Greedy in the Limit* states that while we maintain infinite exploration, we do eventually need to converge to the optimal policy:
  $$\lim_{k\rightarrow\infty} \pi_k(a|s) = \mathbf{1}(a = \argmax_{a' \in \mathcal{A}} Q_k(s, a')).$$

One policy which satisfies the GLIE constraint is simply the hyperbolic
function, parameterized on the episode count $k$:

$$\varepsilon_k = \frac{1}{k}$$

This $\varepsilon$ will decay over time (with every episode) and converge to
zero, satisfying the "Greedy in the Limit" constraint, while staying non-zero
all along, allowing for exploration. The algorithm we have at this stage has the
following update rules:

$$
\begin{align}
  N(S_t, A_t) &\gets N(S_t, A_t) + 1 \\
  N(S_t, A_t) &\gets Q(S_t, A_t) + \frac{1}{N(S_t, A_t)}(G_t - Q(S_t, A_t)) \\
  \varepsilon &\gets 1/k \\
  \pi &\gets \varepsilon-\text{greedy}(Q) \\
\end{align}
$$

This method is called *GLIE Monte-Carlo control*.

### SARSA

The first practical algorithm we want to investigate that performs Q-function
updates is *SARSA*, which stands for the main variables that appear in the
algorithm's equation. Mathematically, this equation can be expressed as

$$Q(S, A) \gets Q(S, A) + \alpha(R + \gamma Q(S', A') - Q(S, A))$$

In this equation, the variables are

- $S$, the current state,
- $A$, an action sampled from the current policy,
- $R$, the reward received for the $(S, A)$ pair,
- $S'$, the next state that we transitioned to,
- $A'$, the next action we sample from the old policy, taken once in state $S'$.

We can now write down an algorithm description for this:

1. Initialize $Q(s, a) \forall s, a$ randomly and set $Q(\text{terminal}, \cdot) = 0$.
2. For each episode:
    1. Pick an initial state $S$,
    2. Choose an initial action $A$ according to the current policy,
    3. For each step of the episode:
        1. Take action $A$ and receive a reward $R$ and observe a new state $S'$.
        2. Sample an action $A'$ from the current policy.
        3. Perform SARSA update: $Q(S, A) \gets Q(S, A) + \alpha(R + \gamma Q(S', A') - Q(S, A))$
        4. $S \gets S', A \gets A'$
    4. If $S$ is terminal, stop.

What is important to note here is that the $\varepsilon$-greedy policy $\pi$ is
actually *implicitly defined in terms of the Q-values*! We pick the greedy
action with probability $1 - \varepsilon$ and a random action otherwise.

We can now make the same generalization of this algorithm as we did for TD
learning, by making the depth of our lookahead variable. For this, let us define
the $n$-step Q-return:

$$q_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_[t+n] + \gamma^n Q(S_{t+n})$$

Then we can define the generalized SARSA algorithm as:

$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha(q_t^{(n)} - Q(S_t, A_t)).$$

Furthermore, just like for the step from TD(0) to $TD(\lambda)$, we now define
$\text{SARSA}(\lambda)$, where we average over all $n$-step Q-returns, giving
each Q-return a weight of $(1-\lambda)\lambda^{n-1}$:

$$q_t^\lambda = (1-\lambda) \sum_{n=1}^\infty$ \lambda^{n-1} q_t^{(n)}.$$

We can then again introduce the idea of an eligibility trace $E_t(s, a)$ that is
incremented on each visit to a state and decays over time:

$$
\begin{align}
  E_0(s, a) &= 0 \\
  E_t(s, a) &= \gamma\lambda E_{t-1}(s, a) + \mathbf{1}(S_t = s, A_t = a)
\end{align}
$$

And then use the same idea of computing the TD error $\delta_t$ and eligibility
trace at each time step and using it to update our Q-values backwards, online:

$$
\begin{align}
  \delta_t &= R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \\
  Q(s, a) &\gets Q(s, a) + \alpha\delta_t E_t(s, a)
\end{align}
$$

Do note that $E_t$ and $Q_s$ have to be updated for every $(\text{state,
action})$ pair, on every iteration! Here is the complete description of the
$\text{SARSA}(\lambda)$ algorithm:

1. Initialize $Q(s, a)$ arbitrarily $\forall s, a$.
2. For each episode:
    1. Initialize the eligibility trace: $E(s, a) = 0 \forall s, a$,
    2. Pick an initial state $S$,
    3. Choose an initial action $A$ according to the current policy,
    4. For each step of the episode:
        1. Take action $A$ and receive a reward $R$ and observe a new state $S'$.
        2. Sample an action $A'$ from the current policy.
        3. Perform the $SARSA(\lambda)$ updates:
          $\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$
          $E(S, A) \gets E(S, A) + 1$
        4. For each $s \in \mathcal{S}$:
            1. For each $a \in \mathcal{A}$:
                1. $Q(s, a) \gets Q(s, a) + \alpha\delta_t E_t(s, a)$
                2. $E(s, a) \gets \gamma\lambda E(s, a)$ # Decay
        5. $S \gets S', A \gets A'$

## Off Policy Learning

*Off Policy* learning is a new class of learning algorithms we want to explore.
The idea here is to not learn the same policy we are using, but to actually
learn from a *different* policy. This is useful, for example, when we want to
learn (from!) the policy of an expert, like a human, or re-use information from
previous policies that we have used in similar contexts. Another idea is to
attempt to learn an *optimal* policy while simultaneously following an
*exploratory* policy.

The first mechanism for off policy learning is *importance sampling*. Importance
sampling is a statistical method for estimating the expectation of a random
variable distributed according to some distribution $P(X)$ that is hard to
sample from, by instead sampling from some other distribution $Q(X)$ according
to the following transformations:

$$
\begin{align}
  \mathbb{E}_{X \sim P}[f(X)] &= \sum P(X) f(X) \\
  &= \sum Q(X)\frac{P(X)}{Q(X)} f(X) \\
  &= \mathbb{E}_{X \sim Q}\left[\frac{P(X)}{Q(X)} f(X)\right]
\end{align}
$$

Notice how by multiplying and dividing by $Q(X)$ we didn't actually change the
equation, but can now sample from $Q(X)$ instead of $P(X)$.

The way we can apply this to our problem setting is by regarding the *behavior*
policy $\mu$ (the prior one we already have) as the distribution we can sample
from easily, like $Q(X)$, and the *target* policy $\pi$ we are learning as
$P(X)$. We can then use the following return:

$$G_t^{\pi/\mu} = \frac{\pi(A_t|S_t)}{\mu(A_t|S_t)}\frac{\pi(A_{t+1}|S_{t+1})}{\mu(A_{t+1}|S_{t+1})}...\frac{\pi(A_t|S_t)}{\mu(A_T|S_T)} \cdot G_t$$

and then use that as our TD target:

$$V(S_t) \gets V(S_t) + \alpha(G_t^{\pi/\mu} - V(S_t))$$

In practice, this has very high variance and does not work well, however.
Instead of this Monte Carlo importance sampling, it is better to use TD learning
with importance sampling. Here, for the case of TD(0), we would perform a
one-step lookahead and then do a single *importance sampling correction*:

$$
V(S_t) \gets V(S_t) + \alpha\left(\frac{\pi(A_t|S_t)}{\mu(A_t|S_t)}(R_{t+1} + \gamma V(S_{t+1})) - V(S_t)\right)
$$

basically using the importance weight $\frac{\pi(A_t|S_t)}{\mu(A_t|S_t)}$ to
correct our TD(0) estimate.

## Q-Learning

Before introducing the Q-learning algorithm, let us understand that for
off-policy learning, we deal with two policies $\mu$ and $\pi$. The former is
the *behavior* policy which we *follow when taking actions*, while the latter is
the *target* policy, which we *learn by*. Now, the idea of Q-learning is similar to SARSA in that in state $S_t$, we sample two actions:

1. $A_t \sim \mu(\cdot | S_t)$ to determine where we go from state $S_t$,
2. $A_{t+1} \sim \mu(\cdot | S_{t+1})$ to determine the next action once in state $S_{t+1}$.

If we just used $A_{t+1}$ to correct our Q-value, we would end up with SARSA:

$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)).$$

The additional step now, which differentiates Q-learning from SARSA and, most
importantly, which makes Q-learning an *off-policy* algorithm, is that we also
sample an additional action $A' \sim \pi(\cdot | S_{t+1})$. Even though it is
the action $A_{t+1}$ we use to progress through the environment, __it is then
$A'$ by which we learn__:

$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1}, A') - Q(S_t, A_t)).$$

So we use the Q-value $Q(S_{t+1}, A')$ of this alternative action $A'$ to update
(correct) our previous Q-value $Q(S_t, A_t)$ instead of the action we progress
by $A_{t+1}$. Intuitively, __this allows us to keep exploring according to
$\mu$, while progressively optimizing all Q-values to follow the target, optimal
policy $\pi$__.

With this background, we can now specialize this description to the actual
Q-learning algorithm. This specialization involves nothing more than assigning
$\mu$ and $\pi$ to concrete policies, which are:

- $\pi$ is chosen as a deterministically greedy policy: $\pi(S_{t+1}) = \argmax_{a'} Q(S_t, a')$, while
- $\mu$ is given an $\varepsilon$-greedy strategy w.r.t. $Q(s, a)$.

This now simplifies the above equations:

$$
\begin{align}
  R_{t+1} + \gamma Q(S_{t+1}, A') &= R_{t+1} + \gamma Q(S_{t+1}, \argmax_{a'} Q(S_t, a'))\\
  &= R_{t+1} + \gamma\max_{a'} Q(S_{t+1}, a')
\end{align}
$$

(since the Q-value of the action that maximizes the Q-value is the maximum
Q-value), giving us the well-known Q-learning update rule:

$$
Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha(R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t))
$$

which will converge to the optimal action-value function $q_\star(s, a)$. Note
again how this is just the Bellman optimality equation for the state-value
function.
