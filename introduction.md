# Reinforcement Learning (Lectures 1-2)

$\newcommand{\argmax}{\mathop{\rm argmax}\nolimits}$

## Introduction

Reinforcement learning is different from supervised learning in that there is no
supervisor, there is only a reward signal. The agent learns on the basis of
trial and error. Also, feedback in Reinforcement Learning may be delayed, not
instantaneous. In RL, time matters. The agent lives in a sequential process,
over time.

### Literature

- http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf
- https://sites.ualberta.ca/~szepesva/RLBook.html

### Definition

RL often involves a scalar *reward* signal $R_t$ that the agent receives at each
time step $t$. There is an agent in an environment, whose goal it is to collect
$R_t$ to maximize the total reward. The flexibility comes from the fact that all
goals (and by extension, all problems) can be described via reward signals and
solved by maximization of expected cumulative reward. This is the *reward
hypothesis*.

The goal in RL is always to select actions to maximize total future rewards. But
actions may have long term consequences, so rewards may be delayed. As such, you
cannot simply be greedy all the time. It may, in fact, be better to sacrifice
immediate reward to gain more long term reward (explore rather than exploit).

The relevant variables at time step $t$ are:

1. An observation $O_t$ an agent makes,
2. An action $A_t$ an agent is able to take,
3. A *scalar* reward signal $R_t$ the agent receives in response to that action.

Next to the agent, there is the environment. The agent takes the action on the
environment, thereby influencing it, and the environment consequently produces
the reward and generates a new observation on which the agent can act (take an
action on).

Note that there really always must be a *scalar* reward, or there is at least
always a conversion from the problem to a scalar reward signal possible.

The *history* is the sequence (or stream) of observations, actions and rewards:

$$H_t = A_1,O_1,R_1, ..., A_t,O_t,R_t.$$

The history is thus all the observable variables up to time step $t$. What
happens next depends on the history. The goal of our algorithm is to map the
history to the next action.

The *state* can capture the history. Formally, *state* is a function of the
history:

$$S_t = f(H_t).$$

There are several parts to the definition of state. The first definition is the
environment state $S_t^e$, which is the environment's internal representation of
the state. There is some set of numbers, some state, within the environment
(e.g. the memory in the Atari emulator), which determines what observation the
environment will emit next. This state is usually not visible to the agent. As
such our algorithm can usually not access the numbers in the environment state.
Interestingly, the environment state may actually contain a lot of noise that
the agent may not care about at all.

Furthermore, there is also the *agent state* $S_t^a$, which captures what
observations and rewards we remember and they were attenuated. Part of
reinforcement learning is to build or find the function $f$, of which the agent
state $S_t^a$ is a function of w.r.t. the history.

Note that if there are several agents, they can be, formally, simply be seen as
part of the environment from the perspective of each individual agent.

### Information State

The Markov property is important for RL. It's definition is that "the future is
independent of the past, given the present". It tells us that the current state
holds just as much information as the history, i.e. it holds all useful
information of the history. Symbolically, we call a state $S_t$ *Markov* iff

$$\Pr[S_{t+1} | S_t] = \Pr[S_{t+1} | S_1,...,S_t].$$

That is, the probability of seeing some next state $S_{t+1}$ given the current
state is exactly equal to the probability of that next state given the entire
history of states. Note that we can always find some Markov state. Though the
smaller the state, the more "valuable" it is. In the worst case, $H_t$ is
Markov, since it represents all known information about itself.

### Fully Observable Environment

The "nicest" kind of environment is one which is *fully observable*, meaning the
agent can *directly* observe the environment state. Formally, this means

$$O_t = S_t^a = S_t^e.$$

Conceptually, this means that the environment has no hidden state that the agent
does not know of. The agent knows everything about the environment and an
observation is made about the environment in its entirety.

This is called a *Markov Decision Process* (*MDP*).

### Partial Observability

It may be that our RL agent has only partial observability. For example, a robot
with camera vision isn't told its absolute location, but only its relative
position (the current image), or a trading agent may only be given current
prices, but not the history. A last example is a poker agent that can only
observe public cards. Now, the agent state is no longer equal to the environment
state!

$$S_t^a \neq S_t^e$$

This is called a *Partially Observable Markov Decision Process* (*POMDP*).

In this case, we must find alternative ways to represent the agent state, since
it can no longer be just the environment state. One possibility is to simply set
the agent state to be the history, i.e. $S_t^a = H_t$. Another alternative is to
have a distribution over the possible states of the environment:

$$S_t^a = (\Pr[S_t^e = s^1], ..., \Pr[S_t^e = s^n]).$$

Finally, another possibility would be to have the state of the agent be the
state of a recurrent neural network:

$$S_t = \sigma(S_{t-1}W_s + O_tW_o)$$

### RL Agents

An RL agent may include one or more of these components:

* *Policy*: An agent's behavior function, how the agent picks an action, given its state.
* *Value function*: A function of how good each state and/or action is.
* *Model*: An agent's representation of the environment.

#### Policy

A policy is the agent's behavior. It is a map from state to action, or from
state to a distribution of actions. For example, with a deterministic policy
$\pi$, we could have $a = \pi(s)$, i.e. the policy always returns the same
action for a certain state. However, we may also have a stochastic
(probabilistic) policy, which defines a probability for each action given a
state:

$$\pi(a | s) = \Pr[A = a | S = s].$$

#### Value Function

The value function is a *prediction of expected future reward*. When we have to
make the choice between two options, the value function tells us which state to
transition to. The value function $v$ depends on a policy $\pi$ (we thus speak
of a "value function for a policy") and tells us the expected reward we will
receive over the next few time steps:

$$v_\pi(s) = \mathbb{E}[R_t + \gamma R_{t+1} + \gamma^2 R_{t + 2} + ... | S_t =
s].$$

As such we want to find the policy that maximizes the value function.

Note how we can have a parameter $\gamma \in [0, 1]$ and decay it over next time
steps, basically saying we care more less ($\gamma < 1$) about the upcoming or
later steps. Since this factor will make future rewards unmeasurable small at
one point, we speak of a *horizon*, i.e. a limit to how far the value function
looks ahead.

#### Models

A model predicts what the environment will do next. There are transition models
and reward models. The transition model $\mathcal{P}$ what the next state of the
environment will be, given the current state (dynamics):

$$P_{ss'}^a = \Pr[S' = s' | S = s, A = a].$$

The reward model $\mathcal{R}$ predicts the next (immediate) reward:

$$R_s^a = \mathbb{E}[R | S = s, A = a].$$

### Taxonomy

We can taxonomize reinforcement learning agents into three basic kinds:

1. *Value based* algorithms have no policy and just a value function.
2. *Policy based* algorithms have no value function and just a policy.
3. *Actor critic* algorithms have both policy and value.

Further more, we can differentiate between *model free* and *model based*
agents.

### Problems solved by Reinforcement Learning

There are two fundamental problems in sequential decision making (making optimal
decisions):

1. First, there is the *reinforcement learning problem*. The environment is
initially unknown. Only by interacting with the environment can the agent learn
about and improve its performance in it.

2. Then, there is the *planning problem*. The model is entirely known. The agent
then performs computations with its model (without any external interaction).
The agent then improves its policy.

The difference is really whether or not the environment is known a priori, or
not.

A key problem of reinforcement learning (in general) is the difference between
*exploration* and *exploitation*. Should the agent sacrifice immediate reward to
explore a (possibly) better opportunity, or should it just exploit its best
possible policy?

Exploration finds more information about the environment. Exploitation exploits
known information to maximize reward. A typical example is whether to always
show the most successful ad in an ad slot, or try a new ad, which may not be
more successful, but may in fact end up bringing more revenue than the old one.

A further distinction is between *prediction* and *control*. Prediction means
evaluating a policy to establish its optimality. Control, in turn, means finding
the best policy. Typically, control requires prediction.

## Markov Decision Processes

Markov Decision Processes (MDPs) formally describe an environment for
reinforcement learning. MDPs are important, since most reinforcement learning
problems can be phrased as MDPs. Naturally, an MDP is Markov and thus the
current state completely characterizes the process.

Markov Processes deal with transitions between states. Such a particular
transition from state $s$ to state $s'$ is defined by a *transition probability*
$\mathcal{P}_{ss'}$, given as

$$\mathcal{P}_{ss'} = \Pr[S_{t+1} = s' | S_t = s].$$

The *state transition matrix* $\mathcal{P}$ defines all such transition
probabilities from any state $s$ to a state $s'$:

$$
\mathcal{P} =
\begin{bmatrix}
  \mathcal{P}_{11} & \dots & \mathcal{P}_{1n} \\
  \vdots & \ddots & \vdots \\
  \mathcal{P}_{n1} & \dots & \mathcal{P}_{nn} \\
\end{bmatrix}
$$

Here, the first index (the rows) is the source state and the second index (the
columns) is the destination state.

To summarize, we can say that a Markov Process is a *memoryless random process*,
i.e. a sequence of random states $S_1, S_2, ...$ that are Markov (have the
Markov property). Formally, we can now represent such an MDP as a pair
$(\mathcal{S, P})$, where $\mathcal{S}$ is the finite set of all possible states
and $\mathcal{P}$ is the transition matrix.

### Markov Reward Process

The next step to go from Markov Decision Processes towards Reinforcement
Learning is a reward signal. For this, we now define a *Markov Reward Process*
(*MRP*), which is characterized by a tuple $(\mathcal{S, P, R, \gamma})$.
$\mathcal{S, P}$ are defined as before, but we now have

1. $\mathcal{R}$, the reward signal, defined as $\mathcal{R}_s = \mathbb{E}[R_{t+1} | S_t = s]$
2. A real discount factor $\gamma \in [0, 1]$ that decays the reward as we look further into the future.

Further, we define a *return* $G_t$ as the total, discounted, reward from
time-step $t$ onwards:

$$G_t = R_{t+1} + \gamma R_{t+2} + ... = \sum_{k=0}^\infty \gamma^k R_{t+1+k}.$$

Note that $G_t$ does not involve any expectation. This is because $G_t$ is not
an estimation of the reward, but the value we would get if we would actually
sample the reward infinitely into the future. One way to interpret the discount
factor $\gamma$ is to say that:

- If $\gamma \approx 0$, we care a lot more about immediate reward,
- If $\gamma \approx 1$ we don't care necessarily when rewards arrive, just about their sum within some time horizon.

In RL we usually do use a discount factor, mainly because of *uncertainty*. If
we are not fully certain that our model of the world is perfect, it is much more
risky to "trust" the model to make accurate predictions far a long the way. We
have more certainty in immediate actions (assuming that uncertainty multiplies
over time). Another reason to have a discount factor is to avoid unbounded
mathematics. You can think of the discount factor as the inverse of an "interest
rate".

Given that we now have the definition of a reward $G_t$, we can redefine a value
function $v$ as:

$$v(s) = \mathbb{E}[G_t | S_t = s].$$

What we can do now is perform some transformations on this equation to get
something that we ultimately know as the *Bellman equation*:

$$
\begin{align}
  v(s) &= \mathbb{E}[G_t | S_t = s] \\
       &= \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t = s] \\
       &= \mathbb{E}[R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + ...) | S_t = s] \\
       &= \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
       &= \mathbb{E}[R_{t+1} + \gamma v(t+1)  | S_t = s] \\
\end{align}
$$

This fundamental knowledge conveys that the expected value at the current state
$s$ is equal to the immediate reward, plus some discounted amount of the
expected reward in the next state. Note here that denoting the reward received
at time step $t$ with $R_{t+1}$ is a notational convention signifying that it is
the reward we would receive from the environment in the next time step, *once we
have engaged with it*.

Now, let us see how we actually compute the value function $v(s)$ in state $s$.
Very simply, this value is given by

$$v(s) = R_{t+1} + \gamma\sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}v(s'),$$

where $\mathcal{S}$ is the set of all states (practically speaking, all states
reachable from $s$) and $\mathcal{P}_{ss'}$ the probability of transitioning
into state $s'$ from our current state $s$. So in this definition, the value
function for a particular state $s$ is equal to the immediate reward, plus a
discounted amount of the weighted sum of the value of all states reachable from
$s$.

There is a much nicer formulation of the Bellman equation with matrices and
vectors. For this, let us define $\mathbf{v} = [v(s_1), ..., v(s_n)]^\top$ as
the vector of all value-function values at time step $t$ and $\mathbf{R} =
[\mathcal{R}_{t+1}^{(1)}, ..., \mathcal{R}_{t+1}^{(n)}]^\top$ as the vector of reward
values in each state at $t$. Then the Bellman equation in vectorized form is
given by

$$\mathbf{v} = \mathbf{R} + \gamma\mathcal{P}\mathbf{v}.$$

Note how the $i$-th component of $\mathcal{P}\mathbf{v}$ will contain $\sum_{s'
\in \mathcal{S}}\mathcal{P}_{s_is'} v(s')$, i.e. precisely the same linear combination
of all states that we could transition into.

What is interesting to note at this point is that the above Bellman equation is
actually just a linear system of equations that we could solve:

$$
\begin{align}
  \mathbf{v} &= \mathbf{R} + \gamma\mathcal{P}\mathbf{v} \\
  \mathbf{v} - \gamma\mathcal{P}\mathbf{v} &= \mathbf{R} \\
  (\mathbf{I} - \gamma\mathcal{P})\mathbf{v} &= \mathbf{R} \\
  \mathbf{v} &= (\mathbf{I} - \gamma\mathcal{P})^{-1}\mathbf{R} \\
\end{align}
$$

where $(\mathbf{I} - \gamma\mathcal{P})^{-1}$ is a matrix inversion and the main
expensive operation. Its computational complexity is $O(n^3)$ for $n$ states and
thus generally infeasible. As such, there exist other methods of evaluating the
value function, including dynamic programming methods, Monte Carlo evaluation
and Temporal-Difference learning.

### Markov Decision Process (2)

So far, we have been discussing Markov Reward Processes. This allows our agent
to be in states, receive rewards for being in a state and estimate the
expected cumulative reward over all future states. However, we do not yet have
the power to actually influence how we get into a particular state, apart from
there being a certain probability. For this reason, Markov Decision Processes
(MDPs) add a set of actions $\mathcal{A}$ to our model, making it a five-tuple:

$$(\mathcal{S, A, P, R, \gamma})$$

Furthermore, since the actions are now what really influence state transitions,
the transition probability matrix $\mathcal{P}$ and reward signal $\mathcal{R}$
are now defined differently:

$$\mathcal{P}_{ss'}^a = \Pr[S_{t+1} = s' | S_t = s, A_t = a]$$
$$\mathcal{R}^a_s = \mathbb{E}[R_{t+1} | S_t = s, A_t = a]$$

basically, transition probabilities and reward signals are now associated with
edges, meaning actions, and not vertices, representing states, in the state
transition diagram. We now get a transition probability $\mathcal{P}_{ss'}^a$
representing the likelihood of moving into state $s'$ given that we are in state
$s$ and take action $a$. Similarly, rewards are now a function of the state
*and* an action.

At this point, it is time to introduce the *policy*. A policy $\pi$ is a
distribution over actions, given some state:

$$\pi(a | s) = \Pr[A_t = a | S_t = s].$$

For now, these policies are stationary, meaning *time-independent* ($A_t \sim
\pi(\cdot|s_t)\,\forall t$). Also, since we are dealing with Markov Processes,
the policy depends only on the state and not on the entire history (since it is
encapsulated by the state).

An interesting fact is that we can always recover an MRP (no actions) from an
MDP (actions). For this, we simply set the transition probability
$\mathcal{P}^\pi_{ss'}$ for the MRP to the average probability of that state
transition across all actions we can take from state $s$:

$$\mathcal{P}^\pi_{ss'} = \sum_{a \in \mathcal{A}} \pi(a|s) P^a_{ss'}$$

and do the same for the reward

$$\mathcal{R}^\pi_s = \sum_{a \in \mathcal{A}} \pi(a|s) R^a_s.$$

#### Value Function for MDPs

We now need two further definitions of the value function for MDPs. The first
value function we define is the *state-value function* $v_\pi(s)$, which is the
expected return starting from state $s$ and strictly following the policy $\pi$:

$$
\begin{align}
  v_\pi(s) &= \mathbb{E}_\pi[G_t | S_t = s] \\
  &= \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s].
\end{align}
$$

This is the value function we have been referring to so far. The second value
function is the *action-value function* $q_\pi(s, a)$, defined as the expected
return starting from state $s$, taking action $a$ and from thereon following
policy $\pi$:

$$
\begin{align}
  q_\pi(s, a) &= \mathbb{E}_\pi[G_t | S_t = s, A_t = a] \\
  &= \mathbb{E}_\pi[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a] \\
\end{align}
.$$

This action-value, also known as "q-value", is very important, as it tells us
directly what action to pick in a particular state.

Given the definition of q-values, we can now better understand the state-value
function as an average over the q-values of all actions we could take in that
state:

$$v_\pi(s) = \sum_{a \in \mathcal{A}}\pi(a|s)q_\pi(s, a)$$

So state-value = average q-value.

We can now also define the action-value in terms of state values (notice the
recursive/recurrent definitions here). Basically, a q-value (action-value) is
equal to the reward $\mathcal{R}_s^a$ that we get from choosing action $a$ in
state $s$, plus a discounted amount of the average state-value of all states we
could end up in given we choose that action:

$$q_\pi(s, a) = \mathcal{R}_s^a + \gamma\sum_{s' \in \mathcal{S}} P_{ss'}^a v_\pi(s')$$

Note how this is quite similar to the definition we had for the state-value in
Markov Reward Processes. So MRP state values are basically like q-values (one
per state, whereas now we have one per action). So we know that state-values are
a weighted sum of q-values and q-values a weighted sum of state-values. Putting
these two things together, we can now get a better intuition of the state-value
of a particular state $s$ as the sum of weighted state-values of all possible
subsequent states $s'$, where the weights are probabilities:

$$
\begin{align}
  v_\pi(s) &= \sum_{a \in \mathcal{A}}\pi(a | s)q_\pi(s, a) \\
  &= \sum_{a \in \mathcal{A}}\pi(a | s)\left(\mathcal{R}_s^a + \gamma\sum_{s' \in \mathcal{S}}\mathcal{P}_{ss'}^a v_\pi(s')\right) \\
\end{align}
$$

in exactly the same way we can define a q-value as a weighted sum of the
q-values of all states we could reach given we pick the action of the q-value:

$$
\begin{align}
q_\pi(s, a) &= \mathcal{R}_s^a + \gamma\sum_{s' \in \mathcal{S}} P_{ss'}^a v_\pi(s') \\
&= \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a'|s')q_\pi(s',a')
\end{align}
$$

At this point, we can define our ultimate goal as finding an optimal state-value
function

$$v_\star(s) = \max_\pi v_\pi(s)$$

or, more importantly, an optimal q-value/action-value function

$$q_\star(s, a) = \max_\pi q_\pi(s, a)$$

i.e. the action-value function using the policy that maximizes the expected
reward. When you have $q_\star(s, a)$, you are done.

One thing you may ask at this point is how value functions can actually be
compared to find a maximum value function. For this to be allowed, we need to
define a partial ordering over policies and value functions:

$$\pi \geq \pi' \Longleftrightarrow v_\pi(s) \geq v_\pi'(s)\, \forall s$$

with emphasis on the $\forall s$, i.e. for all states. Given this ordering, we
can now lay down a theorem that states that:

*For any MDP there exists an optimal policy $\pi_\star$ that is better than or equal to all other policies*, i.e.

$$\exists\pi_\star: \pi_\star \geq \pi, \forall \pi$$

For all such optimal policies (there may be more than one), the corresponding
state-value and action-value functions $v_{\pi_\star}(s) = v_{\star}(s)$ and
$q_{\pi_\star}(s, a) = q_{\star}(s, a)$ are also optimal. What is nice here is
that this means that we only need to find this one optimal policy and we'll
always be able to pick the right action in every state. One way to find this
optimal policy is to maximize over the optimal action-value function:

$$
\pi_\star(a | s) =
\begin{cases}
1 \text{ if } a = \argmax_{a \in \mathcal{A}} q_\star(s, a) \\
0 \text { otherwise }
\end{cases}
$$

This policy would basically assign all the mass of the probability distribution
to the action with the highest q-value, according to the optimal action-value
function, and give other actions a likelihood of 0.

#### Bellman Optimality Equation

While the previous Bellman equations first defined our state-value and q-value
function, the *Bellman optimality equations* actually define how to find the
optimal value functions.

The optimal state-value function $v_\star(s)$ can be found by taking the optimal
action-value function $q_\star$ and for each state, picking the action with the
highest action-value:

$$v_\star(s) = \max_a q_\star(s, a).$$

On the other hand, the optimal action-value function $q_\star$ is found as the
weighted sum of the state-values of all states reached by taking the optimal
action, as determined by the policy $\pi$:

$$q_\star(s, a) = \mathcal{R}_s^a + \gamma\sum_{s \in \mathcal{S}} \mathcal{P}_{ss'}^a v_\star(s).$$

Plugging the Bellman optimality equation for the action-value function into that
of the state-value function gives us a recursive definition of the value
function that we could, theoretically, solve (but approximate in practice):

$$
v_\star(s) = \max_a \left(\mathcal{R}_s^a + \gamma\sum_{s \in \mathcal{S}} \mathcal{P}_{ss'}^a v_\star(s)\right)
$$

and the same for the optimal q-value function:

$$
q_\star(s, a) = \mathcal{R}_s^a + \gamma\sum_{s \in \mathcal{S}} \mathcal{P}_{ss'}^a \max_a q_\star(s, a).
$$

Note how the optimal state-value is determined by the optimal action-value
($\max_a$), while the optimal action-value is given by an *average* (weighted
sum) over all reachable state-values. This is because when we are in a state, we
actually get to pick the action, while once we've picked the action, it is the
transition probabilities (over which we have no control) that determine the
next states and thus the next optimal state-values.
