
# Assignment 2B

## Student Details
**Student Name:** Chaitanya Agrawal  
**Batch No.:** Batch 1

## MULTILAYER PERCEPTRON

**0:** Input and Output values


```python
import numpy as np

X = np.array([[1,0,1,0], [1,0,1,1], [0,1,0,1]])
y = np.array([[1],[1],[0]])
```

##### Table with input and output values

| X |   |   |   |
|---|---|---|---|
| 1 | 0 | 1 | 0 |
| 1 | 0 | 1 | 1 |
| 0 | 1 | 0 | 1 |


| y |
|---|
| 1 |
| 1 |
| 0 |

#### **1:** Initialize weights and biases with random values (There are methods to initialize weights and biases but for now initialize with random values)


```python
# Code to generate random values
wh = np.random.rand(4,3)
bh = np.random.rand(1,3)
wout = np.random.rand(3,1)
bout = np.random.rand(1,1)
# Print weights and bias
print(wh)
print(bh)
print(wout)
print(bout)
```

    [[ 0.17399648  0.48221904  0.27114345]
     [ 0.94624376  0.82994694  0.3027812 ]
     [ 0.20593831  0.64217035  0.00237566]
     [ 0.90988722  0.81834929  0.93616461]]
    [[ 0.18812898  0.67572342  0.63850462]]
    [[ 0.29641655]
     [ 0.51236446]
     [ 0.49083155]]
    [[ 0.06145448]]


##### Table with weights and bias initialized

| wh|   |   |
|---|---|---|
| 0.17399648 | 0.48221904 | 0.27114345]
| 0.94624376 | 0.82994694 | 0.3027812 ]
| 0.20593831 | 0.64217035 | 0.00237566]
| 0.90988722 | 0.81834929 | 0.93616461]]

| bh|   |   |
|---|---|---|
| 0.18812898 | 0.67572342 | 0.63850462 |

| wout |
|------|
| 0.29641655 |
| 0.51236446 |
| 0.49083155 |

| bout |
|------|
| 0.06145448 |

#### **2:** Calculating hidden layer input:


```python
hidden_layer_input = np.dot(X, wh) + bh
# Print hidden_layer_input
print(hidden_layer_input)
```

    [[ 0.56806377  1.80011281  0.91202373]
     [ 1.47795099  2.6184621   1.84818834]
     [ 2.04425995  2.32401965  1.87745043]]


##### Table with hidden layer initialized

| hidden_layer_input|   |   |
|-------------------|---|---|
| 0.56806377 | 1.80011281 | 0.91202373 |
| 1.47795099 | 2.6184621  | 1.84818834 |
| 2.04425995 | 2.32401965 | 1.87745043 |


#### **3:** Performing non-linear transformation on hidden linear input


```python
sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))

hiddenlayer_activations = sigmoid(hidden_layer_input)
# Print hiddenlayer_activations
print(hiddenlayer_activations)
```

    [[ 0.63831628  0.85816267  0.7134141 ]
     [ 0.81426289  0.93204036  0.86391425]
     [ 0.88536633  0.9108469   0.867318  ]]


##### Table with hidden layer activations

| hiddenlayer_activations|   |   |
|------------------------|---|---|
| 0.63831628 | 0.85816267 | 0.7134141  |
| 0.81426289 | 0.93204036 | 0.86391425 |
| 0.88536633 | 0.9108469  | 0.867318   |

#### **4:** Performing linear and non-linear transformation of hidden layer activation at output layer


```python
output_layer_input = np.dot(hiddenlayer_activations, wout) + bout
output = sigmoid(output_layer_input)
# Print output_layer_input & output
print(output_layer_input)
print(output)
```

    [[ 1.04052019]
     [ 1.2043962 ]
     [ 1.21628433]]
    [[ 0.73895036]
     [ 0.76930592]
     [ 0.771409  ]]


##### Table with output

| output     |
|------------|
| 0.73895036 |
| 0.76930592 |
| 0.771409   |

#### **5:** Calculating gradient of Error(E) at output layer


```python
E = y - output
# Print E
print(E)
```

    [[ 0.26104964]
     [ 0.23069408]
     [-0.771409  ]]


##### Table with Error(E)

| E |
|---|
| 0.26104964 |
| 0.23069408 |
| -0.771409  |


#### **6:** Computing slope at output and hidden layer


```python
sigmoid_derivative = lambda x : (sigmoid(x) * (1 - sigmoid(x)))

slope_output_layer = sigmoid_derivative(output)
slope_hidden_layer = sigmoid_derivative(hiddenlayer_activations)
# Print slope_output_layer & slope_hidden_layer
print(slope_output_layer)
print(slope_hidden_layer)
```

    [[ 0.21875368]
     [ 0.21637517]
     [ 0.21620815]]
    [[ 0.22616905  0.20908416  0.22070593]
     [ 0.21274438  0.20269851  0.20859678]
     [ 0.20676386  0.20455709  0.20830754]]


##### Table with Slope_output_layer and Slope_hidden_layer

| Slope_output_layer|
|-------------------|
| 0.21875368 |
| 0.21637517 |
| 0.21620815 |

| Slope_hidden_layer|   |   |
|-------------------|---|---|
| 0.22616905 | 0.20908416 | 0.22070593 |
| 0.21274438 | 0.20269851 | 0.20859678 |
| 0.20676386 | 0.20455709 | 0.20830754 |

#### **7:** Computing delta at output layer


```python
learning_rate = 1.0
d_output = E * slope_output_layer * learning_rate
# Print d_output
print(d_output)
```

    [[ 0.05710557]
     [ 0.04991647]
     [-0.16678491]]


##### Table with d_output

| d_output    |
|-------------|
| 0.05710557  |
| 0.04991647  |
| -0.16678491 |

#### **8:** Calculating Error at hidden layer


```python
error_at_hidden_layer = np.dot(d_output, wout.T)
# Print error_at_hidden_layer
print(error_at_hidden_layer)
```

    [[ 0.01692704  0.02925886  0.02802921]
     [ 0.01479607  0.02557543  0.02450058]
     [-0.04943781 -0.08545466 -0.0818633 ]]


##### Table with error_at_hidden_layer

| error_at_hidden_layer|   |   |
|----------------------|---|---|
| 0.01692704  | 0.02925886  | 0.02802921 |
| 0.01479607  | 0.02557543  | 0.02450058 |
| -0.04943781 | -0.08545466 | -0.0818633 |

#### **9:** Computing delta at hidden layer


```python
d_hiddenlayer = error_at_hidden_layer * slope_hidden_layer
# Print d_hiddenlayer
print(d_hiddenlayer)
```

    [[ 0.00382837  0.00611756  0.00618621]
     [ 0.00314778  0.0051841   0.00511074]
     [-0.01022195 -0.01748036 -0.01705274]]


##### Table with d_hiddenlayer

| d_hiddenlayer|   |   |
|--------------|---|---|
| 0.00382837  | 0.00611756  | 0.00618621  |
| 0.00314778  | 0.0051841   | 0.00511074  |
| -0.01022195 | -0.01748036 | -0.01705274 |

#### **10:** Updating weight at both output and hidden layer


```python
wh = wh + np.dot(X.T, d_hiddenlayer) * learning_rate
wout = wout + np.dot(hiddenlayer_activations.T, d_output) * learning_rate
# Print wout & wh
print(wh)
print(wout)
```

    [[ 0.18097263  0.4935207   0.28244041]
     [ 0.9360218   0.81246658  0.28572846]
     [ 0.21291447  0.65347201  0.01367262]
     [ 0.90281305  0.80605303  0.92422261]]
    [[ 0.22584735]
     [ 0.45597897]
     [ 0.43003947]]


##### Table with weights

| wh|   |   |
|---|---|---|
| 0.18097263 | 0.4935207  | 0.28244041 |
| 0.9360218  | 0.81246658 | 0.28572846 |
| 0.21291447 | 0.65347201 | 0.01367262 |
| 0.90281305 | 0.80605303 | 0.92422261 |

| wout |
|------|
| 0.22584735 |
| 0.45597897 |
| 0.43003947 |


#### **11:** Updating biases at both output and hidden layer


```python
bh = bh + np.sum(d_hiddenlayer, axis=0) * learning_rate
bout = bout + np.sum(d_output, axis=0) * learning_rate
# Print bout & bh
print(bh)
print(bout)
```

    [[ 0.18488318  0.66954473  0.63274883]]
    [[ 0.00169162]]


##### Table with bias

| bout|   |   |
|-----|---|---|
| 0.18488318 | 0.66954473 | 0.63274883 |

| bh |
|----|
| 0.00169162 |

