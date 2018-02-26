# Reinforcement Learning: Value Approximation (Lecture 6)

The methods we have used so far were inherently "tabular", relying on tables for
$V(s)$ or $Q(s, a)$. However, many problems like Go or Backgammon have a very
large state-space, much too large to fit into a table and update efficiently.

The solution is to *approximate* the values of these functions, the basic idea
being that we want to arrive at function approximations $\hat{v}(s; \mathbf{w}),
\hat{q}(s; \mathbf{w})$ that are *parameterized by learnable weights*
$\mathbf{w}$ and best approximate the value function $v_\pi(s)$ and action-value function $q_\pi(s, a)$:

$$\hat{v}(s; \mathbf{w}) \approx v_\pi(s)$$
$$\hat{q}(s, a;\mathbf{w}) \approx q_\pi(s, a)$$

The goal, then, is to find the best possible weights $\mathbf{w}$, either with a
neural network, MC or TD methods. More precisely, there are three rough ways in
which we can use to approximate these two important functions. $\hat{v}(s;
\mathbf{w})$ can be approximated directly from a state $s$. That is, we want our
function approximator (e.g. NN) to tell us how "good" it is to be in state $s$.
For $\hat{q}(s, a; \mathbf{w})$, we have two options. The first is called
"action-in" approximation and approximates the value of a $(\text{state,
action})$ pair. The second approach is called "action-out" and receives only the
current state, while returning not a single value, but a vector $[\hat{q}(s,
a_1; \mathbf{w}), ..., \hat{q}(s, a_m; \mathbf{w})]^\top$ of values with one
entry for each of the $m$ actions we could take from that state $s$.

## Approximation Target

The pre-requisite for approximating the value and action-value function with
neural networks is understanding how we can apply (stochastic) gradient descent
to this problem. For this, let us define $J(\mathbf{w})$ as the differentiable
cost function of our problem, then the gradient of this function is

$$
\nabla_{\mathbf{w}}J(\mathbf{w}) =
\begin{bmatrix}
  \frac{\partial J(\mathbf{w})}{\partial w_1} \\
  \vdots \\
  \frac{\partial J(\mathbf{w})}{\partial w_n} \\
\end{bmatrix}
$$

and we can define the weight update on each iteration of learning as

$$\Delta\mathbf{w} = -\frac{1}{2}\alpha\nabla_{\mathbf{w}}J(\mathbf{w})$$

where $\alpha$ is the learning rate. Wiggling this a little more, we can define
a squared-error term and figure out exactly how we need to update our weights:

$$
\begin{align}
\nabla J(\mathbf{w}) &= -\frac{1}{2}\alpha\nabla_{\mathbf{w}}J(\mathbf{w}) \\
&= -\frac{1}{2}\alpha\nabla_{\mathbf{w}}\mathbb{E}_\pi[(v_\pi(S) - \hat{v}(S; \mathbf{w}))^2] \\
&= \alpha\mathbb{E}_\pi[(v_\pi(S) - \hat{v}(S; \mathbf{w}))\nabla_{\mathbf{w}}\hat{v}(S; \mathbf{w})] \\
\end{align}
$$

such that

$$
\Delta\mathbf{w} = \alpha(v_\pi(S) - \hat{v}(S; \mathbf{w}))\nabla_{\mathbf{w}}\hat{v}(S; \mathbf{w})
$$

where $v_\pi(S)$, for now, is a supervisor value function according to which we
can learn.

## Linear Approximation

Imagine now we have some feature vector $\mathbf{x}$ according to which we want
to learn our value function $\hat{v}(S; \mathbf{w})$. These features would be
distances to objects or the location for a robot, resource usage for a power
plant or any other indications we have for the value of a state. The easiest way
to learn good weights for these features is to do linear regression. That is,
minimize the squared error between our "oracle" value function $v_\pi(S)$ and a
linear combination of our weights and the features. This linear combination
would look like this:

$$
\hat{v}(S; \mathbf{w}) = \mathbf{x}(S)^\top \mathbf{w} = \sum_{j=1}^n x_j(S)w_j
$$

The objective function then becomes

$$J(\mathbf{w}) = \mathbb{E}_\pi[(v_\pi(S) - \mathbf{x}(S)^\top \mathbf{w})^2],$$

which gives us the following update rule:

$$
\begin{align}
\nabla_{\mathbf{w}}\hat{v}(S; \mathbf{w}) &= \mathbf{x}(S) \\
\Delta\mathbf{w} &= \alpha(v_\pi(S) - \hat{v}(S; \mathbf{w}))\nabla_{\mathbf{w}}\mathbf{x}(S)\\
\end{align}
$$

where we make use of the fact that the gradient of a linear combination is just
the coefficients, which here are the features:

$$
\begin{align}
\nabla_{\mathbf{w}} \hat{v}(S; \mathbf{w}) &= \nabla_{\mathbf{w}} \sum_{j=1}^n x_j(S)w_j \\
 &= \sum_{j=1}^n \nabla_{\mathbf{w}}x_j(S)w_j \\
 &= \sum_{j=1}^n \begin{bmatrix}
  \frac{\partial x_j(S)w_j}{\partial w_1} \\ \vdots \\ \frac{\partial x_j(S)w_j}{\partial w_j} \\ \vdots \\ \frac{\partial x_j(S)w_j}{\partial w_n}
\end{bmatrix} \\
 &= \sum_{j=1}^n \begin{bmatrix}
  0 \\ \vdots \\ x_j(S) \\ \vdots \\ 0
\end{bmatrix} \\
&= \mathbf{x}(S)
\end{align}
$$

So in words, the update is

$$
\Delta\mathbf{w} = \text{step-size } \alpha \cdot \text{ prediction error } (v_\pi(S) - \hat{v}(S; \mathbf{w})) \cdot \text { feature values } \mathbf{x}(S)
$$

Note that table lookup (like we used it before) is just a special case where the
features are indicators $x_i = \mathbf{1}(S = s_i)$.

### MC and TD

At  this point, we are still assuming some "supervisor" value function
$v_\pi(S)$ that we can learn from by computing the error between our predictions
$\hat{v}(S; \mathbf{w})$ and the supervisor's "label" value. In practice, we
have no such supervisor function, of course. However, we can go back to what we
learnt about Monte Carlo (MC) and Temporal Difference (TD) learning and generate
a kind of label.

For MC, we would simply treat the returns $G_t$ as the labels

$$
\Delta\mathbf{w} = \alpha(G_t - \hat{v}(S; \mathbf{w}))\nabla_{\mathbf{w}}\hat{v}(S_t; \mathbf{w}),
$$

over time giving us a "dataset"

$$(S_1, G_1), (S_2, G_2), ..., (S_T, G_T).$$

The same could then also be done for TD(0) and TD($\lambda$) approaches:

$$
\Delta\mathbf{w} = \alpha(R_t + \gamma\hat{v}(S_{t+1}; \mathbf{w}) - \hat{v}(S; \mathbf{w}))\nabla_{\mathbf{w}}\hat{v}(S_t; \mathbf{w}),
$$

$$
\Delta\mathbf{w} = \alpha(G_t^{\lambda} - \hat{v}(S; \mathbf{w}))\nabla_{\mathbf{w}}\hat{v}(S_t; \mathbf{w}),
$$

where the difference is, just like for model free control, that MC methods have
higher variance while TD approaches have greater bias. Note that even though we
are "virtually" collecting a dataset over time, we currently do one incremental
update each time.

### $Q$ Approximation

Just as we did in the step from model-based to model-free learning algorithms,
we can now apply this idea of function approximation to the $q_\pi(s, a)$ function
to learn q-values in a model-free way. For this, we would now define our
features as functions of the state *and action*

$$
\mathbf{x}(S, A) =
\begin{bmatrix}
  x_1(S, A) \\
  \vdots \\
  x_n(S, A) \\
\end{bmatrix}
$$

and now, for example, estimate $q_\pi$ via linear regression

$$
\hat{q}(S, A; \mathbf{w}) = \mathbf{x}(S, A)^\top\mathbf{w} = \sum_{j=1}^n x_j(S, A)w_j
$$

and then use updates

$$
\begin{align}
\nabla_{\mathbf{w}}\hat{q}(S, A; \mathbf{w}) &= \mathbf{x}(S, A) \\
\Delta\mathbf{w} &= \alpha(q_\pi(S, A) - \hat{q}(S, A; \mathbf{w}))\mathbf{x}(S, A).
\end{align}
$$

or, as before use MC, TD(0), TD($\lambda$) etc. Note, however, that there is a
possibility when using linear or non-linear TD that the weights explode.
Solutions to this are *gradient TD*, for example: http://www.jmlr.org/proceedings/papers/v24/silver12a/silver12a.pdf

## Batch Methods

So far, we're assuming that we perform one update for every step we make.
However, this is not as efficient and we would thus like to explore *batching*.
This turns reinforcement learning into a kind of short-term, online supervised
learning where to improve our function approximation

$$\hat{v}(s; \mathbf{w}) \approx v_\pi(s)$$

we collect a dataset, or experience, $\mathcal{D}$ consisting of $(\text{state, value})$ pairs

$$
\mathcal{D} = \{(s_1, v_1^\pi), (s_2, v_2^\pi), ..., (s_T, v_T^\pi)\}
$$

and then solve the least-squares problem over $\mathcal{D}$ to optimize our
weights $\mathbf{w}$

$$
\min_{\mathbf{w}} LS(\mathbf{w}) = \min_{\mathbf{w}} \sum_{i=1}^T (v_t^\pi - \hat{v}(s_t; \mathbf{w}))^2.
$$

This dataset $\mathcal{D}$ would eventually just be the full history of all
actions and values we observe during each learning episode. Alternatively, we
can just do stochastic gradient descent (in the original meaning) and sample one
datapoint at a time from our experience

$$(s, v^\pi) \sim \mathcal{D}$$

and apply SGD

$$
\Delta\mathbf{w} = \alpha(v^\pi - \hat{v}(s; \mathbf{w}))\nabla_{\mathbf{w}}(s; \mathbf{w})
$$

which converges to the least squares solution. Another idea would be to do
minibatch gradient descent (?). Note how this differs from our previous
approaches as we are now collecting a dataset over time and sampling random
$(\text{state, action})$ pairs to learn from, rather than just the last one.
This also decorrelates the data, reducing the chance of exploding parameters.
Methods like this one are called *experience replay*.

### Deep Q-Networks (DQN)

The *Deep Q-Network* (*DQN*) architecture published by DeepMind makes use of
precisely the concepts we've looked at so far. They act according to an
$\varepsilon$-greedy policy and store all $(s_t, a_t, r_{t+1}, s_{t+1})$
transitions over time in memory $\mathcal{D}$. At each time-step $t$, they then
sample a mini-batch of transitions from $\mathcal{D}$, using the following loss
function to update the network

$$
\mathcal{L}_i(w_i) = \mathbb{E}_{s,a,r,s'\sim D_i}
\left[\left(r + \gamma \max_a' Q(S', a'; w_i^-) - Q(s, a; w_i)^2\right)\right]
$$

where $w_i^-$ are weights of another neural network. Basically, they keep two
neural networks around, one to produce Q-values according to which to act (like
the behavior policy $\mu$) and one to learn towards (like the target policy
$\pi$). Moreover, the target NN is "frozen" in time a couple thousand steps. As
such, the "label" Q-values actually come from the past. This is again to reduce
the correlation between our current updates and the target, which avoids
explosions in weights when doing TD-learning. The target weights $w_i^-$ are
from this past neural network (that's why the minus).
