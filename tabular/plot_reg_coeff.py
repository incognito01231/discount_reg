from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt

plt_params = {'font.size': 10,
          'lines.linewidth': 2, 'legend.fontsize': 16, 'legend.handlelength': 2,
          'pdf.fonttype':42, 'ps.fonttype':42,
          'axes.labelsize': 18, 'axes.titlesize': 18,
          'xtick.labelsize': 14, 'ytick.labelsize': 14}
plt.rcParams.update(plt_params)

plt.figure()

gammaDivGamma_e = np.linspace(0.1, 0.99, num=1000)
# gammaEval = 0.99
# S = 10
# A = 2
# n_params = S * A
# coeff = (gammaEval - gamma ) / (2 * gamma)
coeff = 0.5*( 1/gammaDivGamma_e - 1)
# coeff /= n_params
plt.plot(gammaDivGamma_e, (coeff))


plt.grid(True)
plt.xlabel(r'$\gamma / \gamma_e$')
plt.ylabel(r'$\lambda = \frac{\gamma_e - \gamma}{2\gamma}$')
# plt.legend()
#
save_PDF = True  # False \ True
if save_PDF:
    plt.savefig('reg_coeff'+ '.pdf', format='pdf', bbox_inches='tight')
else:
    plt.title('Activation Reg. Coeff.')

plt.show()

print('done')
