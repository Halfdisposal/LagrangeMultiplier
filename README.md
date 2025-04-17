![](https://view-counter.tobyhagan.com/?user=Halfdisposal/LagrangeMultiplier)
# Lagrange Multiplier Method 
This method is used to find the extremum of a function subjected to a given constraint.<br>
The general procedure is --<br>
Given a function $f(x, y)$, we want to find its extremas for the constraint $g(x, y) = 0$.<br>
The Lagrange Multiplier,<br>
$$L(x, y) = \Sigma_i \frac{\partial f}{\partial x_i} - \lambda \frac{\partial g}{\partial x_i}$$<br>
in the optimization condition $L(x, y) = 0$<br>
Solving for $L(x, y) = 0$, we get $\frac{\partial f}{\partial x_i} - \lambda \frac{\partial g}{\partial x_i} = 0$ for all *i*<br>

# Using The Code
Clone this repository
```bash
git clone https://github.com/Halfdisposal/LagrangeMultiplier
cd LagrangeMultiplier
```
Install the necessary requirements <br>
```bash
pip install -r requirements.txt
```
Run the main.py file
```bash
python3 main.py
```
In the *main.py* file, the function to optimize is *'f'* and constraint is *'g'*. Play around with different functions. The plot is made with your devices default browser backend.
