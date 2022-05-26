import matplotlib.pyplot as plt
import numpy as np

from layers.weights.initializers import initialize_probabilities_from_expectation
from layers.weights.initializers import initialize_shayer_probabilities_from_expectation

save_figures = True

w_expect_init = np.linspace(-1.5, 1.5, 200)

# Initialize ternary probabilities using the method of Roth [1].
# [1] W. Roth, G. Schindler, H. Fr{\"o}ning, F. Pernkopf;
#     Training Discrete-Valued Neural Networks with Sign Activations Using Weight Distributions; ECML PKDD 2019

q_values = initialize_probabilities_from_expectation(w_expect_init, [-1.0, 0.0, 1.0], q_init_max=0.95)
w_expect = np.sum(q_values * np.asarray([-1.0, 0.0, 1.0]), axis=-1)

f = plt.figure()
ax = f.add_subplot(1, 1, 1)
ax.plot(w_expect_init, q_values[..., 0], 'b')
ax.plot(w_expect_init, q_values[..., 1], 'r')
ax.plot(w_expect_init, q_values[..., 2], 'g')
ax.plot(w_expect_init, w_expect - w_expect_init, 'k-.')
ax.legend(['q(w=-1)', 'q(w=0)', 'q(w=1)', 'Delta E'])
ax.set_title('Roth Initialization: Ternary')
plt.show()
if save_figures:
    f.savefig("init_roth_ternary.png")

# Initialize quinary probabilities using the method of Roth [1].
# [1] W. Roth, G. Schindler, H. Fr{\"o}ning, F. Pernkopf;
#     Training Discrete-Valued Neural Networks with Sign Activations Using Weight Distributions; ECML PKDD 2019

q_values = initialize_probabilities_from_expectation(w_expect_init, [-1.0, -0.5, 0.0, 0.5, 1.0], q_init_max=0.95)
w_expect = np.sum(q_values * np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]), axis=-1)

f = plt.figure()
ax = f.add_subplot(1, 1, 1)
ax.plot(w_expect_init, q_values[..., 0], 'b')
ax.plot(w_expect_init, q_values[..., 1], 'r')
ax.plot(w_expect_init, q_values[..., 2], 'g')
ax.plot(w_expect_init, q_values[..., 3], 'c')
ax.plot(w_expect_init, q_values[..., 4], 'm')
ax.plot(w_expect_init, w_expect - w_expect_init, 'k-.')
ax.legend(['q(w=-1)', 'q(w=-0.5)', 'q(w=0)', 'q(w=0.5)', 'q(w=1)', 'Delta E'])
ax.set_title('Roth Initialization: Quinary')
plt.show()
if save_figures:
    f.savefig("init_roth_quinary.png")

# Initialize ternary probabilities using the method of Shayer [2].
# [2] O. Shayer, D. Levi, E. Fetaya:;
#     Learning Discrete Weights Using the Local Reparameterization Trick; ICLR 2018

q_zro, q_cond_pos = initialize_shayer_probabilities_from_expectation(w_expect_init, w_values=[-1.0, 0.0, 1.0], q_init_max=0.95)
q_pos = (1.0 - q_zro) * q_cond_pos
q_neg = (1.0 - q_zro) * (1.0 - q_cond_pos)
w_expect = q_pos - q_neg

f = plt.figure()
ax = f.add_subplot(1, 1, 1)
ax.plot(w_expect_init, q_neg, 'b')
ax.plot(w_expect_init, q_zro, 'r')
ax.plot(w_expect_init, q_pos, 'g')
ax.plot(w_expect_init, q_cond_pos, 'c--')
ax.plot(w_expect_init, w_expect - w_expect_init, 'k-.')
ax.legend(['q(w=-1)', 'q(w=0)', 'q(w=1)', 'q(w=1|w!=0)', 'Delta E'])
ax.set_title('Shayer Initialization: Ternary')
plt.show()
if save_figures:
    f.savefig("init_shayer_ternary.png")

# Initialize binary probabilities using the method of Shayer [2].
# [2] O. Shayer, D. Levi, E. Fetaya:;
#     Learning Discrete Weights Using the Local Reparameterization Trick; ICLR 2018

q_pos = initialize_shayer_probabilities_from_expectation(w_expect_init, w_values=[-1.0, 1.0], q_init_max=0.95)
w_expect = q_pos - (1.0 - q_pos)

f = plt.figure()
ax = f.add_subplot(1, 1, 1)
ax.plot(w_expect_init, q_pos, 'r--')
ax.plot(w_expect_init, w_expect - w_expect_init, 'k-.')
ax.legend(['q(w=1)', 'Delta E'])
ax.set_title('Shayer Initialization: Binary')
plt.show()
if save_figures:
    f.savefig("init_shayer_binary.png")