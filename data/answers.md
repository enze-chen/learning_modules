# Sample answers to coding exercises

*Author: Enze Chen (University of California, Berkeley)*

Here I provide some sample code that can be used to complete the learning modules. Note that there are many possible solutions!

## Thermodynamics

### Regular_solution_plot
```python
# Function for free energy
def free_energy_curve(x, T, beta=0):
    S_mix = -8.314 * (np.multiply(x, np.log(x)) + np.multiply(1 - x, np.log(1 - x)))
    H_mix = beta * np.multiply(x, 1 - x)
    G_mix = H_mix - T * S_mix
    return G_mix
```
```python
# Generate and plot the curves
fig, ax = plt.subplots()
x = np.linspace(0.001, 0.999, 10000)
T = 1500

for beta in np.linspace(10000, 50000, 6):
    y = free_energy_curve(x, T, beta)
    ax.plot(x, y)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\Delta G$')

plt.show()
```


### Thermo_solution_models
```python
# Creating the phase diagram
for T in Ts: 
    ys = curve_s(x, T) 
    yl = curve_l(x, T) 
    line, idmin, idmax = common_tangent(x, ys, yl, T)
    liquidus.append(x[idmax])
    solidus.append(x[idmin])

# Creating the miscibility gap
for T in T_misc: 
    y = curve_s(x, T)
    line, idmin, idmax = common_tangent(x, y, y, T) 
    solvus.append((x[idmin], T)) 
    solvus.append((x[idmax], T)) 
```


## Characterization

### XRD_indexing

```python
wavelength = 0.154
angles = [43.2531, 50.3844, 74.0626, 89.8768]

df = pd.DataFrame({'Wavelength':wavelength, '2Theta':angles})

df['Theta'] = df['2Theta'] / 2

df['Sine'] = np.sin(np.radians(df['Theta']))

df['Distance'] = df['Wavelength'] / (2 * df['Sine'])

df['Ratio'] = df['Distance^2'][0] / df['Distance^2']
```