# Reinforcement Learning (Lecture 3)

$\newcommand{\argmax}{\mathop{\rm argmax}\nolimits}$

## Dynamic Programming

*Dynamic Programming* deals with dynamic, sequential or temporal problems. In
mathematics, a "program" is not a computer program, but more of a policy, i.e.
guide for an agent. An example would be linear or quadratic programming. The
basic idea behind dynamic programming is to take a problem, break it down into
smaller subproblems that look identical to the original problem, solve those
subproblems, combine them in a way and ultimately solve the overall problem.

A problem must satisfy two criteria to be considered a *dynamic programmming*
problem:

1. It must have an *optimal substructure*.
   This means that the *principle of optimality* applies. Basically, the optimal
   solution should be decomposable into subproblems.
2. The subproblems *overlap*, meaning they recur again and again (so caching helps).

Markov Decision Processes (MDPs) satisfy both of these properties, since they
can be described via a Bellman equation. The Bellman equation is inherently
recursive, satisfying property (1) of optimal substructure. Furthermore, the
value function of our MDP is, in fact, a kind of cache or recombination of past
states.

The two problems we want to solve with Dynamic Programming are *prediction* and
*control*. The prediction problem takes an MDP description and a policy $\pi$
and gives us a value function $v_\pi$ that allows us to calculate the expected
return in a given state. The control problem then involves optimizing this value
function. As such, it gets an MDP description as input and produces an *optimal*
value function $v_star$. The control problem will most likely be a loop, over
all policies, and picking the optimal policy after having evaluated each one.

## Policy Evaluation

The first application of dynamic programming to our reinforcement learning
problem that we want to investigate is *policy evaluation*. The goal of policy
evaluation is to find a better policy given an initial policy $\pi$ (and an MDP
model $(\mathcal{S, A, P, R, \gamma})$). To do so, we use the Bellman
expectation equation for the value function:

$$
\begin{align}
  v_{k+1}(s) &= \sum_{a \in \mathcal{A}} \pi(a|s) q_\pi(s, a) \\
  &= \sum_{a \in \mathcal{A}} \pi(a|s)\left(\mathcal{R}_s^a + \gamma\sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a v_k(s')\right)
\end{align}
$$

In a single update (backup) of the iterative process we would thus, for each
state, evaluate all actions we can pick from that state, and for each state
compute a weighted sum of the *old* state-values of all states reachable when
picking that action.

## Policy Iteration

Now that we have means to evaluate a policy iteratively, we can look into finding an optimal policy. In general, this is composed of two simple steps:

1. Given a policy $\pi$ (initially $\pi_0$), estimate $v_\pi$ via the policy
evaluation algorithm (iterating a fixed number of times or until it stabilizes),
giving you
  $$v_\pi(s) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t = s]$$
2. Generate a new, improved policy $\pi' \geq \pi$ by *greedily* picking
  $$\pi' = \text{greedy}(v_\pi)$$
  Then go back to step (1) to evaluate the policy.

Let's try to understand this deeper. To do so, consider a deterministic policy
$\pi(s) = a$. Then what we are doing in the above two steps is the following:

$$\pi'(s) = \argmax_{a \in \mathcal{A}} q_\pi(s, a)$$

i.e. our new policy will, in each state, pick the action that "gives us the most
q". Now, we can set up the following inequality:

$$
q_\pi(s, \pi'(s)) = \max_{a \in \mathcal{A}} q_\pi(s, a) \geq q_\pi(s, \pi(s)) = v_\pi(s)
$$

which proves that this greedy policy iteration strategy does indeed work, since
the return we get from starting in state $s$, greedily choosing the locally best
action $\argmax_{a \in \mathcal{A}} q(s, a)$ and from thereon following the old
policy $\pi$, must be at least as high as if we had chosen any particular action
$\pi(s)$ and not the optimal one (basically, the maximum of a sequence is at
least as big as any particular value of the sequence).

(Note: I assume the equality in $q_\pi(s, \pi(s)) = v_\pi(s)$ is meant in
expectation?)

What we can show now is that using this greedy strategy not only improves the
next step, but the entire value function. For this, we simply need to do some
expansions inside our definition of the state-value function as a Bellman
expectation equation:

$$
\begin{align}
  v_\pi(s) &\leq q_\pi(s, \pi'(s)) = \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s] \\
&\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1})) | S_t = s] \\
&\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma (R_{t+2} + \gamma^2 v_\pi(S_{t+2})) | S_t = s] \\
&\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 q_\pi(S_{t+2}, \pi'(S_{t+2})) | S_t = s] \\
&\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...)) | S_t = s] \\
&= v_{\pi'}(s)
\end{align}
$$

So in total, we have $v_\pi(s) \leq v_{\pi'}(s)$. Furthermore, if at one point
the policy iteration stabilizes and we have equality in the previous equation

$$
q_\pi(s, \pi'(s)) = \max_{a \in \mathcal{A}} q_\pi(s, a) = q_\pi(s, \pi(s)) = v_\pi(s)
$$

then we also have

$$\max_{a \in \mathcal{A}} q_\pi(s, a) = v_\pi(s)$$

which is precisely the Bellman optimality equation. So at this point, it holds that

$$v_\pi(s) = v_\star(s)\, \forall s \in \mathbb{S}.$$

The last question we must answer is how many steps of policy iteration we should
do to find the optimal policy? Definitely not infinitely many, since we often
notice that the value function stabilizes quite rapidly at some point. So there
are two basic methods:

1. Use $\varepsilon$-convergence, meaning we stop when all values change less than some amount $\varepsilon$ or
2. Just use a fixed number of steps $k$ (thereby introducing another hyperparameter).

## Value Iteration

The next dynamic programming method we want to consider is *value iteration*. In
this case, it is not directly our aim to improve the policy, but rather aims
directly at improving the value function (policy iteration does this as well,
but as a side effect). Basically, while policy iteration iterated on the Bellman
expectation equation, value iteration now iterates on the Bellman *optimality*
equation via the following update rule:

$$
v_\star(s) \gets \max_{a \in \mathcal{A}} \mathcal{R}_s^a + \gamma\sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a v_\star(s')
$$

or, for the step $k \rightarrow k + 1$:

$$
v_{k+1} \gets \max_{a \in \mathcal{A}} \mathcal{R}_s^a + \gamma\sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a v_k(s')
$$

Notice how we assume we already know the solutions to the "subproblems", i.e.
$v_\star(s')$ and then work backwards to find the best solution to the actual
problem (essence of dynamic programming). As such, practically, we can begin
with some initial estimate of the target state-value and then iteratively update
the previous state-values.

Note how value-iteration effectively combines one sweep of policy evaluation,
i.e. one "backup", with one step of policy iteration (improvement), since it
performs a greedy update while also evaluating the current policy. Also, it is
important to understand that the value-iteration algorithm does not require a
policy to work. No actions have to be chosen. Rather, the q-values (rewards +
values of next states) are evaluated to update the state-values. In fact, the
last step of value-iteration is to *output* the optimal policy $\pi^\star$:

$$
\pi^\star(s) = \argmax_a q(s, a) = \argmax_a R_s^a + \gamma\sum_{s' \in \mathcal{S}} P_{ss'}^a v_{\pi^\star}(s')
$$

To derive the above equation, remind yourself of the Bellman optimality equation
for the state-value function:

$$v_\star(s) = \max_a q_\star(s, a)$$

and that for the action-value function:

$$q_\star(s, a) = \mathcal{R}_s^a + \gamma\sum_{s \in \mathcal{S}} \mathcal{P}_{ss'}^a v_\star(s)$$

and plug the latter into the former. Basically, at each time step we will update
the value function for a particular state to be the maximum q value.

## Summary of DP Algorithms (so far)

At this point, we know three DP algorithms that solve two different problems. We
know the *policy evaluation*, *policy iteration* and *value iteration*
algorithms that solve the prediction (1) and control (2, 3) problems,
respectively. Let's briefly summarize each:

1. The goal of __policy evaluation__ is to determine $v_\pi$ given $\pi$. It does so
by starting with some crude estimate $v(s)$ and iteratively evaluating $\pi$.
$v(s)$ is updated to finally give $v_\pi$ via the *Bellman expectation
equation*:
$$
\begin{align}
  v_{k+1} &= \sum_{a \in \mathcal{A}} \pi(a|s) q_\pi(s, a) \\
  &= \sum_{a \in \mathcal{A}}\pi(a | s)\left(R_s^a + \gamma\sum_{s \in \mathcal{S}} P_{ss'}^a v_k(s')\right)
\end{align}
$$

2. __Policy iteration__ combines policy evaluation and the Bellman expectation
equation with updates to the policy. Instead of just updating the state values,
we also update the policy, setting the chosen action in each state to the one
with the highest q-value:
    1. Policy Evaluation,
    2. $\pi' = \text{greedy}(v_\pi)$
Policy iteration algorithms have a time-complexity of $O(mn^2)$ for $m$ actions and $n$ states.
3. Lastly, __Value iteration__ uses the Bellman *optimality* equation to
iteratively update the value of each state to the maximum action-value
attainable from that state
$$
v_{k+1} \gets \max_{a \in \mathcal{A}} q_\pi(s, a) = \max_{a \in \mathcal{A}} \mathcal{R}_s^a + \gamma\sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a v_k(s')
$$
and finally also outputs the optimal policy: $\pi^\star(s) = \argmax_a q(s, a)$
Value iteration algorithms have a time-complexity of $O(m^2n^2)$ for $m$ actions and $n$ states.

## Extensions

One extension to dynamic programming as we have discussed it above is
*asynchronous* DP, where states are updated individually, in any order. This can
significantly improve computation.

The first way to achieve more asynchronous DP is to use *in-place DP*.
Basically, instead of keeping a copy of the old and new value function in each
value-iteration update, you can just update the value functions in-place. The
thing to note here is that asynchronous updates in other parts of the
state-space will directly be affected by this. However, the point is that this
is not actually a bad idea.

An extension of this is *prioritized sweeping*. The idea here is to keep track
of how "effective" or "significant" updates to our state-values are. States
where the updates are more significant are likely further away from converging
to the optimal value. As such, we'd like to update them first. For this, we
would compute this significance, called the *Bellman error*:

$$|v_{k+1}(s) - v_k(s)|$$

and keep these values in a priority queue. You can then efficiently pop the top
of it to always get the state you should update next.

An additional improvement is to do *prioritize local updates*. The idea is that
if your robot is in a particular region of the grid, it is much more important
to update nearby states than faraway ones.

Lastly, in very high dimensional spaces and problems with high branching factor,
it makes sense to sample actions (branches).
