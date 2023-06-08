![Represents the backward pass in a simple feed-forward neural network using the backpropagation algorithm and gradient descent optimization](https://github.com/swayam01/ERA-V1/blob/main/S6/Part-I/Assest/Screenshot%202023-06-09%20at%2012.01.46%20AM.png)

This snippet illustrates the backward pass in a feed-forward neural network, showing the application of the backpropagation algorithm and gradient descent optimization.


---

1. **Forward Propagation**:

- `h1 = w1*i1 + w2*i2` and `h2 = w3*i1 + w4*i2`: These are the outputs of the hidden layer nodes before applying the activation function.
- `a_h1 = σ(h1) = 1/(1 + exp(-h1))` and `a_h2 = σ(h2)`: These are the outputs of the hidden layer nodes after applying the sigmoid activation function (`σ`).
- `o1 = w5*a_h1 + w6*a_h2` and `o2 = w7*a_h1 + w8*a_h2`: These are the outputs of the output layer nodes before applying the activation function.
- `a_o1 = σ(o1)` and `a_o2 = σ(o2)`: These are the outputs of the output layer nodes after applying the sigmoid activation function.
- `E1 = ½ * (t1 - a_o1)²` and `E2 = ½ * (t2 - a_o2)²`: These are the errors for each output node calculated using a mean squared error loss function.
- `E_total = E1 + E2`: This is the total error.

2. **Calculating Gradients for Output Layer Weights (`w5`, `w6`, `w7`, and `w8`)**:

The partial derivative of `E_total` with respect to each output layer weight is calculated using the chain rule. This represents the rate at which the error changes with respect to the corresponding weight.

- `∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1`
- `∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2`
- `∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1`
- `∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2`

3. **Calculating Gradients for Hidden Layer Outputs (`a_h1` and `a_h2`)**:

The partial derivatives of `E1` and `E2` with respect to `a_h1` and `a_h2` are calculated. The total derivatives are the sums of these, representing the rate at which the total error changes with respect to each hidden layer output.

- `∂E_total/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7`
- `∂E_total/∂a_h2 = (a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8`

4. **Calculating Gradients for Hidden Layer Weights (`w1`, `w2`, `w3`, and `w4`)**:

The partial derivatives of `E_total` with respect to each hidden layer weight are calculated using the chain rule.

- `∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1`
- `∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2`
- `∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1`
- `∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2`

These sequence of calculations are used to update the weights of the neural network during training, with the aim of minimizing the total error `E_total`. The weights are updated by subtracting the product of the learning rate and the corresponding derivative from the current weight value.

---

