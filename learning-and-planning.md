# Reinforcement Learning: Integrating Learning and Planning (Lecture 8)

While we previously were interested in learning policies or value functions, we
are now interested in learning *models*. A *model*, in our context, has a quite
specific definition in consisting of a transition model that tells us with what
probability we move from one state to another, and a reward model that tells us
how much reward we get for each action or transition.

## Model-Based RL

Now, in model-free reinforcement learning, we learnt a value function and/or a
policy from experience. In model-*based* RL, we instead learn a model from
experience and then plan a value function.

More precisely, the loop our algorithm follows now starts with some
*experience*. From that experience, we now learn a model of the environment (the
world). Then, we use this model world to do the planning (Q-learning etc.). So,
in a sense, the first step is learning to represent an MDP model, while the
second part is solving that MDP model. The nice thing about learning a model is
that we can learn it with supervised learning.

First, let's try to understand how we model a model. Formally, a model
$\mathcal{M}$ is a representation of an MDP $(\mathcal{S, A, P, R})$
parameterized by $\eta$. For our purposes, we assume the state-space
$\mathcal{S}$ and $\mathcal{A}$ are known. As such, our goal really is to learn
$\mathcal{M} = (\mathcal{P}_{\eta}, \mathcal{R}_{\eta})$ where
$\mathcal{P}_{\eta} \approx \mathcal{P}$ is the state-transition model and
$\mathcal{R}_{\eta} \approx \mathcal{R}$ the reward model defined as

$$
\begin{align}
  S_{t+1} &\sim \mathcal{P}_{\eta}(S_{t+1} | S_t, A_t) \\
  R_{t+1} &= \mathcal{R}_{\eta}(R_{t+1} | S_t, A_t) \\
\end{align}
$$

where we typically assume that rewards and state transitions are distributed
independently, i.e.

$$
\Pr[S_{t+1}, R_{t+1} | S_t, A_t] = \Pr[S_{t+1} | S_t, A_t]\Pr[R_{t+1} | S_t, A_t]
$$

Again, this is really a supervised learning problem. We want to regress the
reward $r = f(s, a)$, while estimating the next state given the current state
and action is a density estimation problem.

### Table Lookup Model

The simplest way in which we can represent a model of the environment is a
*table lookup model*, where we explicitly store values for transition
probabilities and rewards in a table. The way in which we compute these values
is very simple. Basically, every time we make a transition from state $s$,
taking action $a$, we increment a counter $N(s, a)$. The probability for a
transition $s \rightarrow s'$ is then simply the fraction of times that
particular transition occurred relative to all transitions from the source state
$s$. Symbolically, this gives

$$
\hat{\mathcal{P}}_{ss'}^a = \frac{1}{N(s, a)}\sum_{i=1}^T \mathbf{1}(S_t,A_t,S_{t+1} = s,a,s').
$$

We then basically do the same for rewards by simply averaging over all rewards received for a particular transition:

$$
\hat{\mathcal{R}}_s^a = \frac{1}{N(s, a)}\sum_{i=1}^T \mathbf{1}(S_t,A_t = s,a)R_t
$$

### Learning and Planning

Now, let us discuss how we actually combine learning with planning. There are
two approaches for this. Either way, we assume we've learnt a model using the
table lookup model discussed above, or some other model.

For the first approach, we learn our model from real experience and use that
model to do *model-based* learning (planning). Model-based learning gives us a
value function, which gives us a means to compute a policy. It is then this
policy we use to get further real experience to tune our model (and, via more
samples, improve our policy). So we learn a model and then use that model to do
planning. Planning can be done with value iteration, policy iteration, tree
search or other methods.

The other approach combines model learning with model-free planning. In this
case, we only use the model we learn to navigate our environment by sampling
states and rewards, but then do model-free learning with Monte-Carlo control,
SARSA, Q-learning or other methods. So in this case, we basically use the model
we learn to generate synthetic experience which we use to do model-free
learning. Contrast this with the previous idea, where we really learnt from the
model, rather than just using it to navigate the environment. We then use the
state-/action-value functions we learn with model-free learning to navigate the
real environment and collect real experience. This real experience we then use
to update our model with the table-lookup method.

Note that in either case, our approaches are inherently inaccurate, simply
because $(\mathcal{P}_{\eta}, \mathcal{R}_{\eta}) \neq (\mathcal{P, R})$.
Model-based RL is only as good as the model we estimate. Model-free RL will also
only be able to compute a policy that is optimal for our model, but maybe
suboptimal for the real environment.

#### Dyna

Now, there is, in fact, a third approach that is commonly used to combine
learning with planning. This model is called *Dyna* and proposes to combine
learning from real experience as well as sampled experience. As such, it is an
extension of the second approach discussed above. The algorithm for *Dyna-$Q$*
(which uses Q-learning) is the following:

1. Initialize $Q(s, a)$ and $\mathcal{M}(s, a)$ for all $s \in \mathcal{S}, a \in \mathcal{A}$,
2. Forever:
    1. Pick a non-terminal start state $S$,
    2. Sample an action $A$ according to an $\varepsilon$-greedy policy on $Q$,
    3. Take action $A$, observe reward $R$ and state $S'$,
    4. Perform Q-Learning update using real experience:
    $$Q(S, A) \gets Q(S, A) + \alpha(R + \gamma\max_a Q(S', a) - Q(s, a))$$
    5. Update model $\mathcal{M}(s, a)$ using new labels (remember, supervised learning) $S'$ and $R$,
    6. Now, with updated model, repeat (simulate) $n$ times:
        1. $S \gets \text{random state previously seen}$
        2. $A \gets \text{random action previously taken from } S$
        3. Now, use learned model $\mathcal{M}$ to sample $R,S'$
        4. Perform Q-learning update using simulated experience:
        $$Q(S, A) \gets Q(S, A) + \alpha(R + \gamma\max_a Q(S', a) - Q(s, a))$$

There is also a variant called Dyna-$Q^+$ which favors exploration and can
perform better.

## Simulation-Based Search

Now, let us consider in more detail the idea of *simulation-based search*, where
we improve our policy and value-functions via simulations. One general class of
algorithms here is called *forward-search algorithms*. While these methods
operate similar to previous techniques in that they rollout paths in the
action-/state-space, they differ in that they are now rooted at the current
state $S_t$ rather than any previous state, as we described it in the Dyna
algorithm above. So, again, we use our learned model to do simulations and find
out returns from the root state $S_t$ and then use model-free learning
algorithms to improve our policy.

Formally, what we do for simulation-based search is simulate a full episode of experience, starting from *now*, i.e. the state at time $t$:

$$
\{s_t^k, A_t^k, R_{t+1}^k, ..., S_T^k\}_{k=1}^K \sim \mathcal{M}_{\nu}
$$

where $k$ is the index of the current episode, of which we rollout $K$ in total.
Then, there are two basic methods of forward search: Monte-Carlo search and TD
search. We discuss these in more detail below.

### Monte-Carlo Search

In Monte-Carlo search, we rollout full episodes, up to the terminal state. We
again, perform $K$ such rollouts. We then build a *search tree*. What this means
is that for every state $s$ (following the root state $S_t$), we do one rollout.
We then update the Q-values of all states *we've visited so far*. More
precisely, we'll do one rollout from the root state $S_t$ and then set $Q$-value
of that root-state to the ration of wins to rollouts, i.e. for the first rollout
either $1/1$ or $0/1$. Then, we perform *expansion* according to a *simulation
policy*, which tells us how to select an action, e.g. at random or according to
some heuristic. Once we end up in this new state determined by our tree policy,
we do a full new rollout from that new state, $S_{t}^{k+1}$. The important thing
now is that when we get the reward, *we not only update that new root state, but
also all ancestors, up to $S_t$*. Formally, we compute the Q-values of each
$(\text{state, action})$ pair as the average reward

$$
Q(s, a) = \frac{1}{N(s, a)}\sum_{k=1}^K\sum_{u=t}^T \mathbf{1}(S_U,A_u = s,a)G_u
$$

which converges to $q_\pi(s, a)$ in the limit of $K$. Note that next to the
simulation policy, we also simultaneously update a *tree policy*.
The tree policy is very simply the greedy policy

$$a_t \gets \argmax_a Q(s, a)$$

This policy decides what states we follow for *selection* (in MCTS terminology).
Selection is the process of getting from the root state $S_t$ to the next
descendent state that we perform the next rollout from. As such, if we just
start from state $S_t$ and do the first rollout, we'll follow the simulation
policy (expansion). Then, if we get a positive reward from that state, we
continue along that path, meaning for the next rollout we select that state
again and then expand from it. This is because $1/1$ is a better ratio than
$0/0$ for all other states (greedy means no exploration). If the reward was
negative, we follow another action. Note that instead of a greedy policy, we
could also pick an $\varepsilon$-greedy policy for the tree policy.

### TD Search

The difference between TD search and MCTS is again very similar to the
difference between TD learning and MC learning methods. We can again have some
value $\lambda$ which controls how far ahead we bootstrap. Low values of
$\lambda$ will again help to reduce variance. In this case, we update our
action-values with SARSA (for example) instead of reward averaging:

$$\Delta Q(s, a) = \alpha(R + \gamma Q(S', A') - Q(S, A))$$

and then use an $\varepsilon$-greedy policy for selection. We can also use a
function approximation for $Q(s, a)$ (the same is true for MCTS). This is
especially important if the state-space is very large.
