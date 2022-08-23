import numpy as np
from math import exp, sqrt
def f(x, x_dot, sigma_alpha, sigma_gamma, w_u, w_l, alpha, epsilon):
    return np.array([[(((w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]**2*x[1, 0]/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[1, 0]/sigma_alpha**2) + (w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2)*(epsilon + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)))*x_dot[1, 0] + ((w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]**3/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]/sigma_alpha**2) + (w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2)*(epsilon + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)))*x_dot[0, 0])*x_dot[0, 0] + ((4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2)*x[0, 0]*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]*x[1, 0]**2/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)))*x_dot[1, 0] + (4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2)*x[0, 0]*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]**2*x[1, 0]/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)))*x_dot[0, 0])*x_dot[1, 0] - 0.5*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]*x_dot[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]*x_dot[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[1, 0]*x_dot[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]**2*x[1, 0]*x_dot[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2)*x[0, 0]*x[1, 0]*x_dot[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + (w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]*x[1, 0]**2/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]/sigma_alpha**2)*x_dot[1, 0] + (w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2)*(epsilon + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x_dot[1, 0] + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]**2*x[1, 0]*x_dot[0, 0]/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)))*x_dot[1, 0] - 0.5*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]*x_dot[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]*x_dot[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[1, 0]*x_dot[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]**2*x[1, 0]*x_dot[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2)*x[0, 0]*x[1, 0]*x_dot[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + (w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]**3/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]/sigma_alpha**2)*x_dot[0, 0] + (w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2)*(epsilon + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x_dot[0, 0] + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]**2*x[1, 0]*x_dot[1, 0]/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)))*x_dot[0, 0]], [(((w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]*x[1, 0]**2/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]/sigma_alpha**2) + (w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2)*(epsilon + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)))*x_dot[0, 0] + ((w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[1, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[1, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[1, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[1, 0]**3/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[1, 0]/sigma_alpha**2) + (w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2)*(epsilon + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)))*x_dot[1, 0])*x_dot[1, 0] + ((4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2)*x[0, 0]*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]*x[1, 0]**2/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)))*x_dot[1, 0] + (4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[0, 0]/sigma_gamma**2)*x[0, 0]*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]**2*x[1, 0]/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)))*x_dot[0, 0])*x_dot[0, 0] - 0.5*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2*x_dot[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2*x_dot[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]*x_dot[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]*x[1, 0]**2*x_dot[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2)*x[0, 0]*x[1, 0]*x_dot[0, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + (w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[1, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[1, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[1, 0]**3/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[1, 0]**3/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[1, 0]/sigma_alpha**2)*x_dot[1, 0] + (w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2)*(epsilon + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[1, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x_dot[1, 0] + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]*x[1, 0]**2*x_dot[0, 0]/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)))*x_dot[1, 0] - 0.5*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2*x_dot[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]*x[1, 0]**2*x_dot[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]*x_dot[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*x[0, 0]*x[1, 0]**2*x_dot[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*(w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2)*x[0, 0]*x[1, 0]*x_dot[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + (w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*(4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**3*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) + 4*alpha*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**(3/2)) - 2*(1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]**2*x[1, 0]/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)**2) + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]**2*x[1, 0]/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[1, 0]/sigma_alpha**2)*x_dot[0, 0] + (w_l*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2 - w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)*x[1, 0]/sigma_gamma**2)*(epsilon + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x[0, 0]**2/((1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)) + exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2))*x_dot[0, 0] + (1 - exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(w_l*(1 - exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2)) + w_u*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_gamma**2))*exp(-1/2*(x[0, 0]**2 + x[1, 0]**2)/sigma_alpha**2)*x[0, 0]*x[1, 0]**2*x_dot[1, 0]/(sigma_alpha**2*(1 + exp(-2*alpha*sqrt(x[0, 0]**2 + x[1, 0]**2)))**2*(x[0, 0]**2 + x[1, 0]**2)))*x_dot[0, 0]]])