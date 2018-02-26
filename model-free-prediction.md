# Reinforcement Learning: Model Free Prediction (Lecture 4)

## Monte-Carlo Reinforcement Learning

The idea of *Monte-Carlo (MC) Reinforcement Learning* is to learn the value
function directly from experience. For this, the environment must be episodic
and finite, i.e. we have to be able to simulate it to the end. However, we do
not need a model for MC, i.e. no transition probabilities $\mathcal{P}_{ss'}$
and no reward function $\mathcal{R}$. The basic principle is to sample results
and average over them.

There are two basic variants of MC RL. The first is *first-visit* MC and the
second is *every-visit* MC. For each method, we will maintain three statistics
for each state: a counter $N(s)$, an accumulated return $S(s)$ and a mean
expected return $V(s)$, which are updated in the following way:

$$
\begin{align}
  N(s) &\gets N(s) + 1 \\
  S(s) &\gets S(s) + G_t \\
  V(s) &\gets S(s)/N(s) = \frac{1}{N(s)}\sum_t G_t
\end{align}
$$

where we can be assured that $V(s)$ converges to $v_\pi(s)$ (the true
policy/distribution) as $N(s) \rightarrow \infty$ because of the law of large
numbers. The only difference between first-visit and every-visit MC is that if,
in a particular episode, we reach a state more than once, the former method will
only increment the counter once, while the latter will increment it on every
visit.

An important modification we need to make to the update rule for $V(s)$ is to
make the mean incremental. Recall for this that the following transformations
turn the computation of a mean over some quantity $x_i$ into an incremental
update:

$$
\begin{align}
  \mu_t &= \frac{1}{t}\sum_i^t x_i \\
  &= \frac{1}{t}(x_t + \sum_i^{t-1} x_i) \\
  &= \frac{1}{t}(x_t + (t-1)\mu_{t-1}) \\
  &= \frac{x_t}{t} + \frac{t\mu_{t-1}}{t} - \frac{\mu_{t-1}}{t} \\
  &= \frac{t\mu_{t-1}}{t} + \frac{x_t}{t} - \frac{\mu_{t-1}}{t} \\
  &= \mu_{t-1} + \frac{1}{t}(x_t - \mu_{t-1})
\end{align}
$$

where, intuitively, on each time step, we are computing the difference between
our old mean $\mu_{t-1}$ and the new sample $x_t$ and adding a proportional
fraction of that to the old mean. As such, we can rewrite $V(s)$ as an
incremental mean:

$$V(S_t) \gets S(S_t) + \frac{1}{N(S_t)}(G_t - V(S_t))$$

Alternatively, we can "forget" about old episodes and just use a constant update
factor $\alpha$:

$$V(S_t) \gets S(S_t) + \alpha(G_t - V(S_t))$$

also giving us another hyperparameter to tweak. This doesn't just save us one
integer, it also encapsulates the idea of forgetting old information that we may
not necessarily want, for example in non-stationary problems.

Note that the Monte Carlo algorithm is a little different in practice than in
theory. In practice what we will do is assume some policy $\pi$ and then, rather
than do a rollout for each state, do one rollout (episode). Once we've sampled a
single rollout, we can loop through every state we've encountered along the way
and perform the $\text{total return} / \text{number of visits}$ update. To
compute the total return for each state, an efficient method is to loop through
the array backwards and multiply the accumulated reward with $\gamma$ each time
(instead of looping forwards $N$ times, making each update quadratic).

## Temporal Difference Learning

*Temporal Difference (TD) Learning* is another method that learns directly from
experience and is model free. More interestingly and different from Monte Carlo
methods, TD works with *incomplete* episodes via a process known as
*bootstrapping*. Basically, guesses are updated towards further guesses.

In the simplest TD learning algorithm, called TD(0), the update rule for the
value function replaces the expected total return $G_t$ with the immediate
return $R_{t+1} + \gamma V(S_{t+1})$ for the next step:

$$V(S_t) \gets V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$$

where $R_{t+1} + \gamma V(S_{t+1})$ is termed the *TD target* and $\delta_t =
R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ the *TD error*. What this algorithm does
is update the value $V(S_t)$ for state $S_t$ by adding a discounted amount of
the expected return for the next state $S_{t+1}$ that we are just about to go
into. Note that $V(S_{t+1})$ is something we can already compute, simply by
seeing what our current value function estimate $V(s)$ gives us for this next
state $S_{t+1}$.

Intuitively, we are updating our estimate of the value of state $S_t$ by looking
ahead one time step and seeing what the value of the next state is according to
our current value function. Note also that $R_{t+1} + \gamma V(S_{t+1})$ is the
right hand side of the Bellman equation.

The main difference between TD and MC is that TD can learn without knowing the
final outcome. We don't have to complete the entire episode to perform one
update to $V(S_t)$, we only need to look ahead one time step.

Notice also that $R_{t+1} + \gamma V(S_{t+1})$ is a *biased* estimate of
$v_\pi(S_t)$. This is actually one of the other domains where we see major
differences between MC and TD:

- MC has *high variance*, but no *bias*. It therefore always converges.
- TD has *low variance*, but some *bias*. However, it is more efficient.

TD is also more sensitive to the initial value function, since it depends on
itself.

In terms of methods, once more, the difference between TD and MC is that MC
always "rolls out" a whole simulation until it terminates, then takes the real
return to update the estimations of all states on the way back. On the other
hand, TD makes only local lookaheads, moving each state's value estimate closer
to the estimate of the next state. Moreover, the following two things are true
w.r.t. these two methods:

1. MC converges to the solution with minimum squared error, since it updates the
estimates directly with actual returns.
  $$\min \sum_{k=1}^K\sum_{t=1}^{T_k} (g^k_t - V(s_t^k))^2$$
2. TD(0) converges to the *maximum likelihood path through the Markov model that best fits the data*.

(2) is quite remarkable, as TD will actually *find* an MDP representation of the
observations and pick the maximum likelihood path through it. As such, we can
state that TD better exploits the Markov property and is thus better for
problems that are directly explainable as MDPs, while MC will work better in
contexts that aren't always Markov.

Note that one reason why temporal difference learning works only in the forward
direction and not in the backward direction (updating $S_{t+1}$ given $S_t$) is
because on the step from $t$ to $t+1$, we see one step of the "real world", the
real reward and real dynamics (transition probabilities). On the other hand, on
the step backwards, we don't know the reward (since the reward is given for the
action, not the reverse action) (?).

## $TD(\lambda)$

At this point, we have a spectrum of reinforcement learning algorithms with two axes:

1. Whether to do full backups or only sample backups (width of search),
2. Whether to do deep backups or only shallow backups (height/depth of search).

This gives us the following table:

|                | Sample Backup  | Full Backup |
| :------------- | :------------- | :---------- |
| Shallow Backup | TD             | DP          |
| Deep Backup    | MC             | Exhaustive  |

The next class of algorithms is a generalization of temporal difference learning
as we know it right now. This new class is called $TD(\lambda)$.

Before we discuss $TD(\lambda)$, let's start by modifying our original TD
algorithm. The change we make is to introduce a hyperparameter $n$ that controls
how far in the future we look when updating a state (or rather, how far
backwards we look to update prior states). So far, we've encountered the TD(0)
method, where $n = 1$ (note $n$ is not the $0$ in the $TD$). The next step would be to have $n = 2$, where the update equation becomes:

$$V(S_t) \gets V(S_t) + \alpha(R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2}) - V(S_t)))$$

Notice how we now add the reward $R_{t+2}$ of the action taken in time step
$t+1$ instead of the state value, while we now add the value of the state one
step further, at $t+2$. We can generalize this by setting $G_t^{(n)}$ to the
$n$-step lookahead state value:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

where Monte Carlo learning would be equivalent to the $G_t^{(\infty)}$ case. The
update equation then becomes:

$$V(S_t) \gets V(S_t) + \alpha(G_t^{(n)} - V(S_t))).$$

With the luxury of another hyperparameter, we must also ask how to best set this
hyperparameter. The answer is not clear. Instead, the best option is to
*average* over all $G_t^{(1)}, ..., G_t^{(n)}$ up to some chosen $n$, or just
some choices in between, for example

$$\frac{1}{2}G^{(2)} + \frac{1}{2}G^{(4)}.$$

Another solution, which is now this special case of $TD(\lambda)$, is to
introduce yet another hyperparameter $\lambda$ that automatically decays the
value of $G_t^{(k)}$ for greater $k$. The formula for this is

$$G_t^{\lambda} = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1}G_t^{(n)}.$$

which is

- $(1 - \lambda)G_t^{(0)}$ for $n = 1$,
- $(1 - \lambda)G_t^{(0)} + (1 - \lambda)\lambda G_t^{(1)}$ for $n = 2$,
- $(1 - \lambda)G_t^{(0)} + (1 - \lambda)\lambda G_t^{(1)} + (1 - \lambda)\lambda^2 G_t^{(2)}$ for $n = 3$,

and so on. In general the weight for the $n$-th return is $(1 -
\lambda)\lambda^{n-1s}$ Notice that $\lambda$ not only controls the decay of
later returns, it is also a switch for how much we care about closer states and
immediate returns ($\lambda \approx 0$) as opposed to longer lookaheads
($\lambda \approx 1$). For the special case of $\lambda = 0$, we get our TD(0)
case where look ahead exactly one step. The update equation becomes

$$V(S_t) \gets V(S_t) + \alpha \left( G_t^\lambda - V(S_t)\right).$$

Note that we use geometric weighting (a product) because this weighting is
"memory-less" (don't require a counter).

For __online__ learning, where we can't look ahead that far, we can do a
backward form of this algorithm. For this, we keep an *eligibility trace* which
tracks the frequency and recency ("frecency") of a state as $E_t(s)$. It is
defined as:

$$E_0(s) = 0$$
$$E_t(s) = \gamma\lambda E_{t-1}(s) + \mathbf{1}(S_t = s)$$

which increments the trace every time we visit the state and then decays it over time. Then the update is

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
$$V(S) \leftarrow V(s) + \alpha\delta_t E_t(s)$$
