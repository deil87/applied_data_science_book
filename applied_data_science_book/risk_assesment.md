---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Risk assesment:

# Step 1: Define Bayesian Network Model

initialize BayesianNetwork BN with nodes representing risk factors
set conditional probability tables (CPTs) in BN from data or expert knowledge

# Step 2: Infer probabilistic risk exposure

for each relevant risk outcome O:
compute P(O) = BN.infer_probability(O)

# Step 3: Define decision variables and MIP model

initialize MIP model M
declare integer decision variables x_i (e.g., risk treatment choices)

# Step 4: Define objective function incorporating Bayesian probabilities

# Example: minimize expected risk cost = sum over outcomes (cost(O) * P(O))

# risk probabilities depend on decisions x_i which affect risk factors in BN

objective_function = 0
for each outcome O:
\# Assume P(O) can be approximated or parameterized using x_i
P_O_x = function_of(BN, x_i)
objective_function += cost(O) * P_O_x

M.set_objective(minimize, objective_function)

# Step 5: Add constraints related to resources, business rules, and BN structure

M.add_constraints(resource_limits)
M.add_constraints(logical_constraints_on_x)
M.add_constraints(relationship_constraints_inferred_from_BN)

# Step 6: Solve MIP for optimal decisions

optimal_solution = M.solve()

# Step 7: Extract and interpret results

report optimal_solution.x_i as risk treatment decisions
report implied risk probabilities from BN given x_i</q> Now give me toy example with numbers. Show me how uncertainties in the risk are incorporated into MIP model or used as constraint there.

Here is a toy example demonstrating how uncertainties from a Bayesian Network (BN) can be incorporated into a Mixed Integer Programming (MIP) model, using simple numbers and constraints.

Assume we want to optimize two risk treatment decisions $x_1, x_2$ which are binary variables (0 = don't treat, 1 = treat). There are two risk outcomes $O_1, O_2$ with costs \$1000 and \$500 respectively.

***

### Bayesian Network Probabilities (simplified)

- Without treatment:
$P(O_1) = 0.4$, $P(O_2) = 0.3$
- Treatment decreases risk by 50%:
$P(O_1 | x_1=1) = 0.2$, $P(O_2 | x_2=1) = 0.15$

***

### Objective: Minimize expected risk cost plus treatment costs

- Treatment costs:
$C_{\text{treat}_1} = 200$, $C_{\text{treat}_2} = 150$
- Expected risk cost:

$$
\text{Expected Cost} = 1000 \times P(O_1 | x_1) + 500 \times P(O_2 | x_2) + 200 \times x_1 + 150 \times x_2
$$

Using a big-M approach in MIP, model probabilities as functions of treatment variables:

$$
P(O_1 | x_1) = 0.4 - 0.2 x_1
$$

$$
P(O_2 | x_2) = 0.3 - 0.15 x_2
$$

### MIP formulation:

Minimize

$$
Z = 1000 (0.4 - 0.2 x_1) + 500 (0.3 - 0.15 x_2) + 200 x_1 + 150 x_2
$$

Subject to

$$
x_1 \in \{0,1\}, \quad x_2 \in \{0,1\}
$$

***

### Interpretation:

- If $x_1 = 0$, risk $O_1$ has 0.4 probability, if $x_1=1$, reduced to 0.2; similar for $x_2$.
- Treatment variables are integer decision variables in MIP.
- The objective combines risk costs weighted by conditional probabilities from BN and treatment costs.
- Constraints can be added to impose budget limits or logical rules.

***

### Summary of flow:

1. Bayesian network provides probabilistic risk reductions based on decisions.
2. Probabilities expressed as linear functions of integer decisions.
3. MIP minimizes total expected cost balancing risk and treatment cost.
4. Solving the MIP yields optimal treatment decisions considering uncertainty.

This toy example illustrates integrating uncertainties from Bayesian networks into MIP by parameterizing risk probabilities as linear expressions of decision variables, enabling risk-aware optimization under uncertainty.[^4][^5][^6]
