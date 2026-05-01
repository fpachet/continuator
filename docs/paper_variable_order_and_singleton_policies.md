# Variable Order, Positional Constraints, and Singleton Avoidance

This note summarizes the design discussion around the new context-BP
Continuator. It is intended as source material for a paper rather than as API
documentation.

## Problem

The Continuator learns a variable-order Markov model over symbolic viewpoints.
At generation time, it must satisfy positional constraints while preserving the
musical advantage of high-order contexts: high orders give stylistic
specificity, but they can also copy the training corpus too literally or become
incompatible with future constraints.

This creates three related questions:

1. How should the model choose the effective order at each generation step?
2. How should positional constraints be propagated through the model?
3. When does the resulting sampler have an exact probabilistic interpretation?

The new context-BP design separates these concerns:

- Sparse context graphs represent the learned continuation structure.
- Belief propagation computes feasibility and conditional weights under
  positional constraints.
- An explicit order policy chooses among feasible context orders.

## Exact Fixed-Order Baseline

For a fixed order `K`, a Markov model can be rewritten as a first-order model
whose state is the current context:

```text
(x_{t-K+1}, ..., x_t) -- emit y --> (x_{t-K+2}, ..., x_t, y)
```

This gives a sparse context-state graph. Positional constraints are unary
constraints on the emitted edge labels. Standard forward-backward or BP on this
finite graph gives exact marginals and an exact conditional sampler for:

```text
P(x_1:T | prefix, constraints, K)
```

In this setting, the sampling probability at a step is proportional to:

```text
transition_weight(edge) * backward_message(next_state, remaining_positions)
```

This is the clean theoretical baseline.

## Variable Order Under Constraints

A variable-order model must decide which suffix length to use. In the current
implementation, BP backward messages are prepared for orders from `Kmax` down
to 1. At each generated position, the sampler builds candidate sets from the
longest currently available context down to shorter suffixes. A candidate edge
is retained only if:

1. its emitted symbol satisfies the local positional constraint, and
2. the BP backward message says that the edge can still lead to a complete
   sequence satisfying all future constraints.

Thus constraints are propagated before the order policy is applied.

This does not by itself define a unique probability law over all variable-order
models. Instead, it creates a family of BP-aware sequential samplers, depending
on the chosen order policy.

## Longest Feasible Policy

The simplest policy is:

```text
try the longest feasible context;
if no feasible continuation exists, back off by one order;
repeat down to order 1.
```

This is easy to explain and close to the fixed-order BP baseline. It has an
important practical property: backoff occurs only when required by the learned
model or by future constraints.

However, it is still a policy, not the same thing as exact conditioning of a
single globally specified variable-order distribution, unless that distribution
is explicitly defined to include this deterministic backoff rule.

## Singleton Avoidance

The classic Continuator contains an additional anti-copying heuristic. If a
higher-order context has only one continuation, using it often reproduces a
fragment of the training corpus almost literally. The classic strategy usually
skips such singleton contexts and tries a shorter suffix instead.

In the current context-BP implementation, this is represented by
`SingletonAvoidingBackoffPolicy`. After BP and constraints have filtered the
candidates, a singleton higher-order context is accepted with probability:

```text
1 / (order + 1)
```

Otherwise it is skipped. The skipped symbol is also suppressed at intermediate
lower orders when alternatives exist. Order 1 remains the final fallback, so
the sampler can still proceed if the skipped symbol is forced by the model or
by the constraints.

This reproduces the musical intuition of the original system:

```text
prefer high order,
but avoid using high order merely to copy a unique continuation.
```

## What Remains Exact

Singleton avoidance changes the distribution being sampled. Therefore the
result is not exact sampling from the unmodified fixed-order context graph, nor
from an automatically selected maximum-likelihood variable-order model.

But the constraint propagation remains exact relative to each tested context
graph:

- A candidate edge is considered only if it can complete the remaining
  constrained sequence.
- The probabilities among the retained candidates are still weighted by the
  BP backward messages.
- If the policy removes candidates, the sampler renormalizes over the remaining
  BP-feasible candidates.

So the correct statement is:

```text
The sampler performs exact BP-based feasibility and weighting inside each
candidate context graph, then applies a musically motivated order-selection
policy before sampling.
```

For the singleton policy, the resulting process is best understood as a
sequential constrained generation policy rather than as exact conditioning of a
pure Markov model. If desired, it could be given a formal probability law by
including the policy's stochastic skip/accept decisions as part of the
generative process.

## Optimal Variable-Order Models

There is a classical statistical literature on learning "optimal" variable
orders. The central object is usually not one global order but a context tree:
different histories may use different suffix lengths.

A naive maximum-likelihood criterion is insufficient. If all higher-order
contexts are allowed, training likelihood generally improves as the model
memorizes more context. Without a complexity penalty, MLE tends to overfit.

Classical solutions include:

- Rissanen's Context algorithm and MDL-style model selection.
- Penalized maximum likelihood or BIC for context-tree pruning.
- Context Tree Weighting, which mixes over many context trees rather than
  selecting a single one.
- Probabilistic suffix trees, which keep longer suffixes only when their
  continuation distributions differ meaningfully from shorter suffixes.

A penalized-likelihood context-tree model would define a cleaner variable-order
Markov source:

```text
training corpus
-> suffix counts up to Kmax
-> context-tree pruning by MDL/BIC/statistical tests
-> one selected context for each history
-> exact BP conditioning on the selected context graph
```

This gives a stronger theoretical story. It answers the question "which order
should be used?" during model estimation rather than during generation.

## Why The Practical Policy May Be Preferable Here

The Continuator is not only a prediction or compression model. It is an
interactive constrained generation system. The statistically optimal context
tree for predicting the corpus is not necessarily the best musical policy for
producing a varied continuation under user constraints.

In practice, the singleton policy has several advantages:

- It is transparent: one can explain exactly why the model backed off.
- It is controllable: the skip probability can be exposed as a musical
  parameter.
- It reduces literal copying from high-order singleton contexts.
- It still uses BP to avoid choices that would make future constraints
  impossible.
- It preserves the Continuator's historical behavior while improving the
  constraint mechanism.

Thus the paper can present two levels:

1. A formal level: sparse context graphs with exact BP for positional
   constraints.
2. A practical musical level: an order policy, especially singleton avoidance,
   applied after BP feasibility propagation.

## Possible Paper Wording

The following wording is deliberately conservative:

> We represent each bounded-order context model as a sparse context-state graph
> and use belief propagation to compute exact feasibility and conditional
> transition weights under positional constraints. Variable-order behavior is
> introduced by an explicit order-selection policy that compares feasible
> candidate contexts from longest to shortest. The default musical policy
> inherits the original Continuator's singleton-avoidance strategy: high-order
> contexts with a unique continuation are usually skipped, unless required by
> the constraints. This policy sacrifices the interpretation of the sampler as
> exact conditioning of a single fixed Markov source, but preserves exact
> constraint propagation inside each candidate graph and provides a transparent
> mechanism for avoiding literal copying.

An alternative, more theoretical wording:

> A fully probabilistic variant would first learn or weight a variable-order
> context tree using penalized likelihood, MDL, or context-tree weighting, and
> would then condition the resulting source model on positional constraints.
> In the present Continuator setting, we favor an explicit generation-time
> policy because it is musically interpretable and directly controls the
> tradeoff between specificity and variation.

## References

- Rissanen, J. (1983). A universal data compression system. IEEE Transactions
  on Information Theory, 29(5), 656-664.
  https://doi.org/10.1109/TIT.1983.1056741
- Buhlmann, P., & Wyner, A. J. (1999). Variable length Markov chains. Annals
  of Statistics, 27(2), 480-513. Technical report:
  https://statistics.berkeley.edu/tech-reports/479
- Willems, F. M. J., Shtarkov, Y. M., & Tjalkens, T. J. (1995). The
  context-tree weighting method: basic properties. IEEE Transactions on
  Information Theory, 41(3), 653-664.
  https://doi.org/10.1109/18.382012
- Ron, D., Singer, Y., & Tishby, N. (1996). The power of amnesia: Learning
  probabilistic automata with variable memory length. Machine Learning,
  25(2-3), 117-149. https://doi.org/10.1007/BF00114008
