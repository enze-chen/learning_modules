# Sample answers to coding exercises

*Author: Enze Chen (University of California, Berkeley)*

Here I provide some sample code that can be used to complete the learning modules. Note that there are many possible solutions!


## Computational

### Cellular_automaton_models
```python
# Initialize parameters for recrystallization
L = 50
n = 5
idn = n
```
```python
# Initialization function
def initialize(L, n):
    cells = np.zeros(shape=(L, L))
    np.put(a=cells, \
           ind=np.random.default_rng().choice(a=L**2, size=n, replace=False), \
           v=range(1, idn+1))
    return cells
```
```python
# Get neighboring states
def neighbors(arr, i, j):
    L = len(arr)
    neighbor_list = []
    for k in range(i - 1, i + 2):
        for l in range(j - 1, j + 2):
            neighbor_list.append(arr[k % L, l % L])
    neighbor_list.pop(4)
    return np.array(neighbor_list)
```
```python
# Evolve the CA
def evolve(arr, n):
    # Grow existing grains
    temp = arr.copy()
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            if temp[i, j] == 0:
                neighbor_list = neighbors(arr=temp, i=i, j=j)
                if np.sum(neighbor_list):
                    mode = scipy.stats.mode(neighbor_list[neighbor_list != 0])
                    arr[i, j] = mode[0]
                    
    # Nucleate new embryos
    L = len(arr)
    k = np.random.default_rng().choice(L, n)
    l = np.random.default_rng().choice(L, n)
    for i in range(n):
        if arr[k[i], l[i]] == 0:
            idn += 1
            arr[k[i], l[i]] = idn
    
    return arr
```
```python
# Initialize paramters for spinodal decomposition
A = 1.5
D = 0.7
N = 100
```
```python
# Compute average of neighbors
def avg_neighbors(arr, i, j):
    m, n = arr.shape
    fnn = (arr[(i - 1) % m, j] + arr[(i + 1) % m, j] + arr[i, (j - 1) % n] + \
           arr[i, (j+1)%n]) / 6
    snn = (arr[(i - 1) % m, (j - 1) % n] + arr[(i - 1) % m, (j + 1) % n] + \
           arr[(i + 1) % m, (j - 1) % n] + arr[(i + 1) % m, (j + 1) % n]) / 12
    return fnn + snn
```
```python
# Update function for spinodal decomposition animation
def update(dummy):    
    temp_cells = np.zeros(cells.shape)
    for i in range(temp_cells.shape[0]):
        for j in range(temp_cells.shape[1]):
            temp_cells[i, j] = A * np.tanh(cells[i, j]) + \
                               D * (avg_neighbors(cells, i, j) - cells[i, j])

    for i in range(temp_cells.shape[0]):
        for j in range(temp_cells.shape[1]):
            cells[i, j] = temp_cells[i, j] - avg_neighbors(temp_cells - cells, i, j) 
    
    img = ax.imshow(cells)
    return (img,)
```



### Monte_Carlo_Ising_model
```python
# Set constants
J = 1
k_B = 1

# Create the spins
def create_spins(L):
    return np.random.choice([-1, 1], size=(L, L))
```
```python
# Plotting utility
def plot_spins(spins, T):
    n = len(spins)
    fig, ax = plt.subplots()
    ax.imshow(spins)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.set_title(f'{n} x {n} lattice, T = {T}')
    plt.show()
```

```python
# Compute the total energy and magnetization
def compute_energy(spins):
    a, b = spins.shape
    energy = 0
    for i in range(a):
        for j in range(b):
            energy -= spins[i, j] * spins[(i + 1) % a, j] + \
                      spins[i, j] * spins[i, (j + 1) % b]
    return energy / 2

def compute_mag(spins):
    return np.sum(spins)
```

```python
# Perform a single MC sweep
def mc_sweep(spins, beta):
    n = len(spins)
    for _ in range(n):
        for _ in range(n):
            # Randomly choose the site
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            
            # Hacky way to quickly compute the energy change
            nb_sum = spins[(i + 1) % n, j] + spins[(i - 1) % n, j] + \
                     spins[i, (j + 1) % n] + spins[i, (j - 1) % n]
            dE = 2 * spins[i, j] * nb_sum
            
            # Acceptance criteria for Metroplis-Hastings algorithm
            # If dE < 0 this is always true and we're done
            if np.random.random() < np.exp(-dE * beta):
                spins[i, j] *= -1

    return spins
```

```python
# Function to perform the MC simulation
def mc_ising_model(L, Ts, eqsteps, mcsteps):    
    # Store the final values as a function of temperature
    E_T = []
    M_T = []
    C_T = []
    X_T = []
    
    for T in Ts:
        # Initialize everything for each temperature
        spins = create_spins(L)
        beta = 1 / T
        E = E_sq = M = M_sq = 0
        
        # Perform equilibration MC runs
        for _ in range(eqsteps):
            mc_sweep(spins, beta)

        # Perform additional MC runs
        for _ in range(mcsteps):
            mc_sweep(spins, beta)
            Ex = compute_energy(spins)
            Mx = compute_mag(spins)
            E += Ex
            E_sq += Ex**2
            M += Mx
            M_sq += Mx**2
    
        # Add quantities to the lists
        E_T.append(E / L**2 / mcsteps)
        M_T.append(M / L**2 / mcsteps)
        C_T.append((E_sq/mcsteps - E**2/mcsteps**2) / L**2 * beta**2)
        X_T.append((M_sq/mcsteps - M**2/mcsteps**2) / L**2 * beta)
    
    # Return the four lists as a tuple
    return (E_T, M_T, C_T, X_T)
```

```python
# Experimental parameters for the MC run
L = 4
Ts = np.linspace(1, 5, 9)
eqsteps = 1800
mcsteps = 200
```
```python
# Plotting the results
# Estimate the critical temperature
Tc_est = np.mean([Ts[np.argmax(C_T)], Ts[np.argmax(X_T)]])
print(f'Tc estimate: {Tc_est}')

# Plot the properties as a function of T
fig, ax = plt.subplots(nrows=1, ncols=4)
ax[0].plot(Ts, E_T, 'o', c='C0')
ax[0].set_title('Energy')
ax[0].set_xlabel(r'$T$')

ax[1].plot(Ts, np.absolute(M_T), 'o', c='C1')
ax[1].set_title('Magnetization')
ax[1].set_xlabel(r'$T$')

ax[2].plot(Ts, C_T, 'ro', c='C2')
ax[2].set_title('Heat capacity')
ax[2].set_xlabel(r'$T$')

ax[3].plot(Ts, X_T, 'go', c='C3')
ax[3].set_title('Susceptibility')
ax[3].set_xlabel(r'$T$')
plt.show()
```


### Machine_learning_Ising_model
```python
# Gathering the data
labels = pd.read_csv(labelurl, header=None).to_numpy().ravel()
temps = pd.read_csv(tempsurl, header=None).to_numpy().ravel()
```

```python
# Splitting the data
T_low, T_high = (2.00, 2.50)
crit_ind = (temps >= T_low) & (temps <= T_high)
safe_ind = (temps < T_low) | (temps > T_high)

X_data = data[safe_ind, :]
y_data = labels[safe_ind]
X_test = data[crit_ind, :]
y_test = labels[crit_ind]

X_train, X_val, y_train, y_val = \
    train_test_split(X_data, y_data, test_size=0.3, shuffle=True)
```

```python
# Initialize and train the dummy classifier
dummy_clf = DummyClassifier(strategy='prior', random_state=seed)
dummy_clf.fit(X_train, y_train)

# Print accuracy of dummy predictions 
print(f'The accuracy on the training set is {dummy_clf.score(X_train, y_train):.4f}')
print(f'The accuracy on the validation set is {dummy_clf.score(X_val, y_val):.4f}')
print(f'The accuracy on the test set is {dummy_clf.score(X_test, y_test):.4f}')
```

```python
# Initialize and train the logistic regression classifier
lr_clf = LogisticRegression(max_iter=1e4, random_state=seed)
lr_clf.fit(X_train, y_train)

# Print accuracy of logistic regression predictions
print(f'The accuracy on the training set is {lr_clf.score(X_train, y_train):.4f}')
print(f'The accuracy on the validation set is {lr_clf.score(X_val, y_val):.4f}')
print(f'The accuracy on the test set is {lr_clf.score(X_test, y_test):.4f}')
```
```python
# Calculating the probabilities and uncertainties
Ts = np.unique(temps)
for T in Ts:
    ind = (temps == T)
    probs = lr_clf.predict_proba(data[ind, :])
    means = np.mean(probs, axis=0)
    stds = np.std(probs, axis=0)
    mean_dis_lr.append(means[0])
    mean_ord_lr.append(means[1])
    err_dis_lr.append(stds[0])
    err_ord_lr.append(stds[1])
```

```python
# Initialize and train the perceptron
mlp_clf = MLPClassifier(hidden_layer_sizes=(20,), random_state=seed)
mlp_clf.fit(X_train, y_train)

# Print accuracy of perceptron predictions
print(f'The accuracy on the training set is {mlp_clf.score(X_train, y_train):.4f}')
print(f'The accuracy on the validation set is {mlp_clf.score(X_val, y_val):.4f}')
print(f'The accuracy on the test set is {mlp_clf.score(X_test, y_test):.4f}')
```
```python
# Calculating the probabilities and uncertainties
Ts = np.unique(temps)
for T in Ts:
    ind = (temps == T)
    probs = mlp_clf.predict_proba(data[ind, :])
    means = np.mean(probs, axis=0)
    stds = np.std(probs, axis=0)
    mean_dis_mlp.append(means[0])
    mean_ord_mlp.append(means[1])
    err_dis_mlp.append(stds[0])
    err_ord_mlp.append(stds[1])
```


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
betas = np.linspace(10000, 50000, 6)

for beta in betas:
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
```
```python
# Creating the miscibility gap
for T in T_misc: 
    y = curve_s(x, T, beta_s)
    line, idmin, idmax = common_tangent(x, y, y, T, beta_s) 
    solvus.append((x[idmin], T)) 
    solvus.append((x[idmax], T)) 
```



## Characterization

### XRD_indexing

```python
# Experimental data
wavelength = 0.154
angles = [43.2531, 50.3844, 74.0626, 89.8768]

# Creating the DataFrame
df = pd.DataFrame({'Wavelength':wavelength, '2Theta':angles})
```
```python
# Obtaining the angles
df['Theta'] = df['2Theta'] / 2

# Computing sin(theta)
df['Sine'] = np.sin(np.radians(df['Theta']))
```
```python
# Computing the interplanar distance
df['Distance'] = df['Wavelength'] / (2 * df['Sine'])

# Computing the ratios between the planes
df['Ratio'] = df['Distance^2'][0] / df['Distance^2']
```

### XRD_plotting

```python
def compute_thetas(planes, a, wavelength):
    thetas = np.arcsin(wavelength / (2 * a / np.linalg.norm(planes, axis=1)))
    return thetas
```
```python
def compute_F(planes, basis, f):
    F = f * np.sum(np.exp(2j * np.pi * (planes @ basis.T)), axis=1)
    return F
```
```python
def compute_m(planes):
    m = np.array([2 ** np.count_nonzero(p) * \
                  len(set(itertools.permutations(p))) for p in planes])
    return m
```
```python
def compute_Lp(thetas):
    Lp = np.divide(1 + np.cos(2 * thetas) ** 2,
                   np.multiply(np.sin(thetas) ** 2, np.cos(thetas)))
    return Lp
```
```python
# Putting it all together
angles = 2 * np.degrees(thetas)
intensities = np.multiply(np.abs(compute_F(planes, basis, f)) ** 2, 
                          np.multiply(compute_m(planes), compute_Lp(thetas)))
```