from navierstokes import *
np.seterr(over='raise')

def main():
    exercise1()
    exercise2()
    exercise3()
    exercise4()
    extras()
    return

def exercise1():
    Lx=2
    Ly=2

    dx=0.01
    dy=0.01
    dt=10**(-3)

    U=5

    driven_cavity=NewtonianFluid(Lx, Ly, dx, dy, dt, U=U)

    driven_cavity.set_boundary_conditions_p(BVC_up=0, BVC_down="Neumann", BVC_left="Neumann", BVC_right="Neumann")
    driven_cavity.set_boundary_conditions_u(BVC_up=U, BVC_down=0, BVC_left=0, BVC_right=0)
    driven_cavity.set_boundary_conditions_v(BVC_up=0, BVC_down=0, BVC_left=0, BVC_right=0)

    # ### 1 ###
    # print("\n1")
    # driven_cavity.set_parameters(dx= 0.05, dy=0.05, dt=10**(-3), mu=0.1, rho=0.01, U=5)
    # U=5
    # driven_cavity.set_boundary_conditions_p(BVC_up=0, BVC_down="Neumann", BVC_left="Neumann", BVC_right="Neumann")
    # driven_cavity.set_boundary_conditions_u(BVC_up=U, BVC_down=0, BVC_left=0, BVC_right=0)
    # driven_cavity.set_boundary_conditions_v(BVC_up=0, BVC_down=0, BVC_left=0, BVC_right=0)
    # stability=driven_cavity.check_stability()
    # driven_cavity.simulate_system()
    # driven_cavity.plot_quiver_speed()
    # driven_cavity.plot_quiver_pressure()
    # driven_cavity.plot_streamline()
    # driven_cavity.plot_vorticity()
    # driven_cavity.plot_vorticity_symlog()
    # driven_cavity.plot_u_profile()
    # driven_cavity.plot_v_profile()
    # driven_cavity.plot_profiles_adami()

    # ### 2 ###
    # print("\n2a")
    # driven_cavity.set_parameters(dx= 0.030, dy=0.030, dt=10**(-3), mu=0.1, rho=0.01)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system(max_iterations=10**5)
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")

    # print("\n2b")
    # driven_cavity.set_parameters(dx= 0.05, dy=0.05, dt=0.00275, mu=0.1, rho=0.01, U=5)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system()
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")

    # ### 3 ###
    # print('\n3')
    # h_list=[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.5]
    # for h in h_list:
    #     driven_cavity.set_parameters(dx= h, dy=h, dt=10**(-3), mu=0.1, rho=0.01, U=5)
    #     stability=driven_cavity.check_stability()
    #     try:
    #         driven_cavity.simulate_system(max_iterations=10**6)
    #         print(f"For h={driven_cavity.dx}, it takes {driven_cavity.iterations} iterations to reach convergence.\n")
    #     except:
    #         print("Error, overflow!")

    # driven_cavity.set_parameters(dx=0.2, dy=0.2, dt=10**(-3), mu=0.1, rho=0.01, U=5)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system()
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")


    # ### 4 ###
    # print("\n4")
    # dt_list=[2.5*10**-3, 1*10**-3, 7.5*10**-4, 5*10**-4, 2.5*10**-4, 1*10**-4, 7.5*10**-5, 5*10**-5, 2.5*10**-5, 1*10**-5]
    # for dt in dt_list:
    #     driven_cavity.set_parameters(dx= 0.05, dy=0.05, dt=dt, mu=0.1, rho=0.01, U=5)
    #     stability=driven_cavity.check_stability()
    #     try:
    #         driven_cavity.simulate_system(max_iterations=10**8)
    #         print(f"For dt={driven_cavity.dt}, it takes {driven_cavity.iterations} iterations to reach convergence.\n")
    #     except:
    #         print("Error, overflow!")

    # driven_cavity.set_parameters(dx=0.05, dy=0.05, dt=10**(-5), mu=0.1, rho=0.01, U=5)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system(max_iterations=10**8)
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")

    # ### 5 ###
    # print("\n5")
    # driven_cavity.set_parameters(dx= 0.05, dy=0.05, dt=10**(-3), mu=0.1, rho=0.01, U=5)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system(tol=10**(-15), max_iterations=10**6)
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")

    # ### 6 ###
    # print("\n6a")
    # U=5
    # driven_cavity.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=1, rho=1, U=U)
    # driven_cavity.set_boundary_conditions_u(BVC_up=U, BVC_down=0, BVC_left=0, BVC_right=0)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system(max_iterations=10**5)
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")

    # print("\n6b")
    # U=1
    # driven_cavity.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=0.2, rho=1, U=U)
    # driven_cavity.set_boundary_conditions_u(BVC_up=U, BVC_down=0, BVC_left=0, BVC_right=0)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system(max_iterations=10**5)
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")

    # ### 7 ###
    # print("\n7a")
    # U=5
    # driven_cavity.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=0.2, rho=1, U=U)
    # driven_cavity.set_boundary_conditions_u(BVC_up=U, BVC_down=0, BVC_left=0, BVC_right=0)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system(max_iterations=10**5)
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")

    # print("\n7b")
    # U=5
    # driven_cavity.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=0.08, rho=1, U=U)
    # driven_cavity.set_boundary_conditions_u(BVC_up=U, BVC_down=0, BVC_left=0, BVC_right=0)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system(max_iterations=10**5)
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")


    # ### 8 ###
    # print("\n8")
    # U=7
    # driven_cavity.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=0.05, rho=1, U=U)
    # driven_cavity.set_boundary_conditions_u(BVC_up=U, BVC_down=0, BVC_left=0, BVC_right=0)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system(max_iterations=10**6)
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")

    # ### 9 ###
    # print("\n9")
    # U=1
    # driven_cavity.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=1.15, rho=1, U=U)
    # driven_cavity.set_boundary_conditions_u(BVC_up=U, BVC_down=0, BVC_left=0, BVC_right=0)
    # stability=driven_cavity.check_stability()
    # try:
    #     driven_cavity.simulate_system(max_iterations=10**6)
    #     driven_cavity.plot_subplots()
    #     driven_cavity.plot_profiles()
    # except:
    #     print("Error, overflow!")
    return

def exercise2():
    Lx=2
    Ly=1

    dx=0.01
    dy=0.01
    dt=10**(-3)

    U=5

    laminar_flow=NewtonianFluid(Lx, Ly, dx, dy, dt)

    laminar_flow.set_boundary_conditions_p(BVC_up="Neumann", BVC_down="Neumann", BVC_left="Periodic", BVC_right="Periodic")
    laminar_flow.set_boundary_conditions_u(BVC_up=0, BVC_down=0, BVC_left="Periodic", BVC_right="Periodic")
    laminar_flow.set_boundary_conditions_v(BVC_up=0, BVC_down=0, BVC_left="Periodic", BVC_right="Periodic")

    # ### 1 ###
    # F=10
    # mu=0.5
    # laminar_flow.set_parameters(dx= 0.05, dy=0.05, dt=10**(-3), mu=mu, rho=1, U=0, Fx=F)
    # try:
    #     laminar_flow.simulate_system(max_iterations=10**5)
    #     laminar_flow.plot_subplots()
    #     laminar_flow.plot_profiles()
    #     print(f"The Reynolds number is equal to: {(laminar_flow.Ly*laminar_flow.rho*np.average(np.sqrt(laminar_flow.u**2+laminar_flow.v**2)))/(laminar_flow.mu)}")
    # except:
    #     print("Error, overflow!")


    # ### 2 ###
    # F=100
    # mu=0.5
    # laminar_flow.set_parameters(dx= 0.05, dy=0.05, dt=10**(-3), mu=mu, rho=1, U=0, Fx=F)
    # try:
    #     laminar_flow.simulate_system(max_iterations=10**5)
    #     laminar_flow.plot_subplots()
    #     laminar_flow.plot_profiles()
    #     print(f"The Reynolds number is equal to: {(laminar_flow.Ly*laminar_flow.rho*np.average(np.sqrt(laminar_flow.u**2+laminar_flow.v**2)))/(laminar_flow.mu)}")
    # except:
    #     print("Error, overflow!")


    # ### 3 ###
    # F=10
    # mu=0.05
    # laminar_flow.set_parameters(dx= 0.05, dy=0.05, dt=10**(-3), mu=mu, rho=1, U=0, Fx=F)
    # try:
    #     laminar_flow.simulate_system(max_iterations=10**5)
    #     laminar_flow.plot_subplots()
    #     laminar_flow.plot_profiles()
    #     print(f"The Reynolds number is equal to: {(laminar_flow.Ly*laminar_flow.rho*np.average(np.sqrt(laminar_flow.u**2+laminar_flow.v**2)))/(laminar_flow.mu)}")
    # except:
    #     print("Error, overflow!")


    # F=10
    # mu=0.005
    # laminar_flow.set_parameters(dx= 0.05, dy=0.05, dt=10**(-3), mu=mu, rho=1, U=0, Fx=F)
    # try:
    #     laminar_flow.simulate_system(max_iterations=10**5)
    #     laminar_flow.plot_subplots()
    #     laminar_flow.plot_profiles()
    #     print(f"The Reynolds number is equal to: {(laminar_flow.Ly*laminar_flow.rho*np.average(np.sqrt(laminar_flow.u**2+laminar_flow.v**2)))/(laminar_flow.mu)}")
    # except:
    #     print("Error, overflow!")

    # F=10
    # mu=1.28
    # laminar_flow.set_parameters(dx= 0.05, dy=0.05, dt=10**(-3), mu=mu, rho=1, U=0, Fx=F)
    # try:
    #     laminar_flow.simulate_system(max_iterations=10**5)
    #     laminar_flow.plot_subplots()
    #     laminar_flow.plot_profiles()
    #     print(f"The Reynolds number is equal to: {(laminar_flow.Ly*laminar_flow.rho*np.average(np.sqrt(laminar_flow.u**2+laminar_flow.v**2)))/(laminar_flow.mu)}")
    # except:
    #     print("Error, overflow!")
    return

def exercise3():
    Lx=3
    Ly=1

    dx=0.01
    dy=0.01
    dt=10**(-3)

    U=5

    laminar_flow_obstacle=NewtonianFluid(Lx, Ly, dx, dy, dt)

    laminar_flow_obstacle.set_boundary_conditions_p(BVC_up="Neumann", BVC_down="Neumann", BVC_left="Periodic", BVC_right="Periodic")
    laminar_flow_obstacle.set_boundary_conditions_u(BVC_up=0, BVC_down=0, BVC_left="Periodic", BVC_right="Periodic")
    laminar_flow_obstacle.set_boundary_conditions_v(BVC_up=0, BVC_down=0, BVC_left="Periodic", BVC_right="Periodic")

    obstacle=laminar_flow_obstacle.add_obstacle()
    obstacle.set_dimensions(x0=0.7, x1=1.3, y0=0, y1=0.2)

    # ### 1 ###
    # F=1
    # mu=0.1
    # laminar_flow_obstacle.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_obstacle.simulate_system(max_iterations=10**5)
    # laminar_flow_obstacle.plot_subplots()
    # laminar_flow_obstacle.plot_u_profile(0.6)
    # laminar_flow_obstacle.plot_u_profile(1.0)
    # laminar_flow_obstacle.plot_u_profile(1.4)
    # laminar_flow_obstacle.plot_v_profile(0.1)
    # laminar_flow_obstacle.plot_v_profile(0.3)
    # laminar_flow_obstacle.plot_v_profile(0.7)
    # print(f"The Reynolds number is equal to: {(laminar_flow_obstacle.Ly*laminar_flow_obstacle.rho*np.average(np.sqrt(laminar_flow_obstacle.u**2+laminar_flow_obstacle.v**2)))/(laminar_flow_obstacle.mu)}")

    # ### 2 ###
    # F=50
    # mu=0.1
    # laminar_flow_obstacle.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_obstacle.simulate_system(max_iterations=10**5)
    # laminar_flow_obstacle.plot_subplots()
    # laminar_flow_obstacle.plot_u_profile(0.6)
    # laminar_flow_obstacle.plot_u_profile(1.0)
    # laminar_flow_obstacle.plot_u_profile(1.4)
    # laminar_flow_obstacle.plot_v_profile(0.1)
    # laminar_flow_obstacle.plot_v_profile(0.3)
    # laminar_flow_obstacle.plot_v_profile(0.7)
    # print(f"The Reynolds number is equal to: {(laminar_flow_obstacle.Ly*laminar_flow_obstacle.rho*np.average(np.sqrt(laminar_flow_obstacle.u**2+laminar_flow_obstacle.v**2)))/(laminar_flow_obstacle.mu)}")

    # F=205
    # mu=0.1
    # laminar_flow_obstacle.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_obstacle.simulate_system(max_iterations=10**5)
    # laminar_flow_obstacle.plot_subplots()
    # print(f"The Reynolds number is equal to: {(laminar_flow_obstacle.Ly*laminar_flow_obstacle.rho*np.average(np.sqrt(laminar_flow_obstacle.u**2+laminar_flow_obstacle.v**2)))/(laminar_flow_obstacle.mu)}")

    # ### 3 ###
    # F=5
    # mu=1.15
    # laminar_flow_obstacle.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_obstacle.simulate_system(max_iterations=10**5)
    # laminar_flow_obstacle.plot_subplots()
    # print(f"The Reynolds number is equal to: {(laminar_flow_obstacle.Ly*laminar_flow_obstacle.rho*np.average(np.sqrt(laminar_flow_obstacle.u**2+laminar_flow_obstacle.v**2)))/(laminar_flow_obstacle.mu)}")

    # F=5
    # mu=0.05
    # laminar_flow_obstacle.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_obstacle.simulate_system(max_iterations=10**5)
    # laminar_flow_obstacle.plot_subplots()
    # print(f"The Reynolds number is equal to: {(laminar_flow_obstacle.Ly*laminar_flow_obstacle.rho*np.average(np.sqrt(laminar_flow_obstacle.u**2+laminar_flow_obstacle.v**2)))/(laminar_flow_obstacle.mu)}")

    # F=5
    # mu=0.02
    # laminar_flow_obstacle.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_obstacle.simulate_system(max_iterations=10**5)
    # laminar_flow_obstacle.plot_subplots()
    # print(f"The Reynolds number is equal to: {(laminar_flow_obstacle.Ly*laminar_flow_obstacle.rho*np.average(np.sqrt(laminar_flow_obstacle.u**2+laminar_flow_obstacle.v**2)))/(laminar_flow_obstacle.mu)}")

    # ### 4 ###
    # F=20
    # mu=0.1
    # obstacle.set_dimensions(x0=0.7, x1=1.3, y0=0, y1=0.4)
    # laminar_flow_obstacle.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_obstacle.simulate_system(max_iterations=10**5)
    # laminar_flow_obstacle.update_reynolds_number()
    # laminar_flow_obstacle.plot_subplots()


    # F=20
    # mu=0.1
    # obstacle.set_dimensions(x0=0.7, x1=1.3, y0=0, y1=0.6)
    # laminar_flow_obstacle.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_obstacle.simulate_system(max_iterations=10**5)
    # laminar_flow_obstacle.update_reynolds_number()
    # laminar_flow_obstacle.plot_subplots()


    # F=20
    # mu=0.1
    # obstacle.set_dimensions(x0=0.7, x1=1.3, y0=0, y1=0.8)
    # laminar_flow_obstacle.set_parameters(dx= 0.03, dy=0.03, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_obstacle.simulate_system(max_iterations=10**5)
    # laminar_flow_obstacle.update_reynolds_number()
    # laminar_flow_obstacle.plot_subplots()

    return

def exercise4():
    Lx=2
    Ly=1

    dx=0.01
    dy=0.01
    dt=10**(-3)

    U=5

    laminar_flow_trough=NewtonianFluid(Lx, Ly, dx, dy, dt)

    laminar_flow_trough.set_boundary_conditions_p(BVC_up="Neumann", BVC_down="Neumann", BVC_left="Periodic", BVC_right="Periodic")
    laminar_flow_trough.set_boundary_conditions_u(BVC_up=0, BVC_down=0, BVC_left="Periodic", BVC_right="Periodic")
    laminar_flow_trough.set_boundary_conditions_v(BVC_up=0, BVC_down=0, BVC_left="Periodic", BVC_right="Periodic")

    obstacle1=laminar_flow_trough.add_obstacle()
    obstacle1.set_dimensions(x0=0, x1=0.7, y0=0, y1=0.2)
    obstacle2=laminar_flow_trough.add_obstacle()
    obstacle2.set_dimensions(x0=1.3, x1=2, y0=0, y1=0.2)

    # ### 1 ###
    # print('\n1')
    # F=0.1
    # mu=0.2
    # laminar_flow_trough.set_parameters(dx= 0.02, dy=0.02, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_trough.simulate_system(max_iterations=10**5)
    # laminar_flow_trough.plot_subplots()
    # print(f"The Reynolds number is equal to: {(laminar_flow_trough.Ly*laminar_flow_trough.rho*np.average(np.sqrt(laminar_flow_trough.u**2+laminar_flow_trough.v**2)))/(laminar_flow_trough.mu)}")

    # ### 2 ###
    # print('\n2a')
    # F=50
    # mu=0.2
    # laminar_flow_trough.set_parameters(dx= 0.02, dy=0.02, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_trough.simulate_system(max_iterations=10**5)
    # laminar_flow_trough.plot_subplots()
    # print(f"The Reynolds number is equal to: {(laminar_flow_trough.Ly*laminar_flow_trough.rho*np.average(np.sqrt(laminar_flow_trough.u**2+laminar_flow_trough.v**2)))/(laminar_flow_trough.mu)}")

    # print('\n2b')
    # F=150
    # mu=0.2
    # laminar_flow_trough.set_parameters(dx= 0.02, dy=0.02, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_trough.simulate_system(max_iterations=10**5)
    # laminar_flow_trough.plot_subplots()
    # print(f"The Reynolds number is equal to: {(laminar_flow_trough.Ly*laminar_flow_trough.rho*np.average(np.sqrt(laminar_flow_trough.u**2+laminar_flow_trough.v**2)))/(laminar_flow_trough.mu)}")

    # ### 3 ###
    # print('\n3a')
    # F=10
    # mu=0.12
    # laminar_flow_trough.set_parameters(dx= 0.02, dy=0.02, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_trough.simulate_system(max_iterations=10**5)
    # laminar_flow_trough.plot_subplots()
    # print(f"The Reynolds number is equal to: {(laminar_flow_trough.Ly*laminar_flow_trough.rho*np.average(np.sqrt(laminar_flow_trough.u**2+laminar_flow_trough.v**2)))/(laminar_flow_trough.mu)}")

    # print('\n3b')
    # F=10
    # mu=0.05
    # laminar_flow_trough.set_parameters(dx= 0.02, dy=0.02, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_trough.simulate_system(max_iterations=10**5)
    # laminar_flow_trough.plot_subplots()
    # laminar_flow_trough.plot_u_profile(0.75)
    # laminar_flow_trough.plot_u_profile(1.0)
    # laminar_flow_trough.plot_u_profile(1.25)
    # laminar_flow_trough.plot_v_profile(0.1)
    # laminar_flow_trough.plot_v_profile(0.25)
    # laminar_flow_trough.plot_v_profile(0.5)
    # print(f"The Reynolds number is equal to: {(laminar_flow_trough.Ly*laminar_flow_trough.rho*np.average(np.sqrt(laminar_flow_trough.u**2+laminar_flow_trough.v**2)))/(laminar_flow_trough.mu)}")

    # ### extra ###
    # print('\n Extra')
    # obstacle1.set_dimensions(x0=0, x1=0.2, y0=0, y1=0.5)
    # obstacle2.set_dimensions(x0=1.8, x1=2, y0=0, y1=0.5)
    # F=50
    # mu=0.06
    # laminar_flow_trough.set_parameters(dx= 0.01, dy=0.01, dt=10**(-4), mu=mu, rho=1, U=0, Fx=F)
    # laminar_flow_trough.simulate_system(max_iterations=5*10**5)
    # laminar_flow_trough.plot_subplots()
    # print(f"The Reynolds number is equal to: {(laminar_flow_trough.Ly*laminar_flow_trough.rho*np.average(np.sqrt(laminar_flow_trough.u**2+laminar_flow_trough.v**2)))/(laminar_flow_trough.mu)}")
    return

def extras():

    # ### Plane wing  ###
    Lx=2
    Ly=2

    dx=0.01
    dy=0.01
    dt=10**(-3)

    # U=0

    # plane=NewtonianFluid(Lx, Ly, dx, dy, dt, U=U)

    # plane.set_boundary_conditions_p(BVC_up="Periodic", BVC_down="Periodic", BVC_left="Periodic", BVC_right="Periodic")
    # plane.set_boundary_conditions_u(BVC_up="Periodic", BVC_down="Periodic", BVC_left="Periodic", BVC_right="Periodic")
    # plane.set_boundary_conditions_v(BVC_up="Periodic", BVC_down="Periodic", BVC_left="Periodic", BVC_right="Periodic")
    # plane.set_parameters(dx= 0.02, dy=0.02, dt=10**(-4), mu=0.2, rho=1, U=U, Fx=30, Fy=15)
    # wing1=plane.add_obstacle()
    # wing1.set_dimensions(0.7, 1.0, 0.9, 1.1)
    # wing2=plane.add_obstacle()
    # wing2.set_dimensions(1.0, 1.2, 0.9, 1.05)
    # wing3=plane.add_obstacle()
    # wing3.set_dimensions(1.2, 1.4, 0.9, 0.95)
    # wing4=plane.add_obstacle()
    # wing4.set_dimensions(0.6, 0.7, 0.9, 1.0)
    # plane.simulate_system(max_iterations=10**5)
    # plane.plot_subplots()

    # ### Driven cavity plus ###

    # U=20

    # driven_cavity_plus=NewtonianFluid(Lx, Ly, dx, dy, dt, U=U)
    # driven_cavity_plus.set_boundary_conditions_p(BVC_up="Neumann", BVC_down="Neumann", BVC_left="Neumann", BVC_right="Neumann")
    # driven_cavity_plus.set_boundary_conditions_u(BVC_up=0, BVC_down=0, BVC_left=0, BVC_right=0)
    # driven_cavity_plus.set_boundary_conditions_v(BVC_up=0, BVC_down=0, BVC_left=0, BVC_right=0)
    # driven_cavity_plus.set_parameters(dx= 0.02, dy=0.02, dt=10**(-4), mu=0.3, rho=1, U=U)
    # belt1=driven_cavity_plus.add_obstacle()
    # belt1.set_dimensions(0.5, 0.9, 0.5, 0.9)
    # belt1.set_boundary_conditions_p(0, 0, 0, 0)
    # belt1.set_boundary_conditions_u(U, -U, 0, 0)
    # belt1.set_boundary_conditions_v(0, 0, U, -U)

    # belt2=driven_cavity_plus.add_obstacle()
    # belt2.set_dimensions(0.5, 0.9, 1.1, 1.5)
    # belt2.set_boundary_conditions_p(0, 0, 0, 0)
    # belt2.set_boundary_conditions_u(-U, U, 0, 0)
    # belt2.set_boundary_conditions_v(0, 0, -U, U)

    # belt3=driven_cavity_plus.add_obstacle()
    # belt3.set_dimensions(1.1, 1.5, 1.1, 1.5)
    # belt3.set_boundary_conditions_p(0, 0, 0, 0)
    # belt3.set_boundary_conditions_u(U, -U, 0, 0)
    # belt3.set_boundary_conditions_v(0, 0, U, -U)
    
    # belt4=driven_cavity_plus.add_obstacle()
    # belt4.set_dimensions(1.1, 1.5, 0.5, 0.9)
    # belt4.set_boundary_conditions_p(0, 0, 0, 0)
    # belt4.set_boundary_conditions_u(-U, U, 0, 0)
    # belt4.set_boundary_conditions_v(0, 0, -U, U)
    # driven_cavity_plus.simulate_system(max_iterations=10**4)
    # driven_cavity_plus.plot_subplots()

    # ### Garden hose ###

    # Lx=3
    # Ly=1

    # dx=0.01
    # dy=0.01
    # dt=10**(-3)

    # U=5

    # garden_hose=NewtonianFluid(Lx, Ly, dx, dy, dt)

    # garden_hose.set_boundary_conditions_p(BVC_up="Neumann", BVC_down="Neumann", BVC_left="Periodic", BVC_right="Periodic")
    # garden_hose.set_boundary_conditions_u(BVC_up=0, BVC_down=0, BVC_left="Periodic", BVC_right="Periodic")
    # garden_hose.set_boundary_conditions_v(BVC_up=0, BVC_down=0, BVC_left="Periodic", BVC_right="Periodic")

    # obstacle1=garden_hose.add_obstacle()
    # obstacle1.set_dimensions(x0=0.9, x1=1.2, y0=0, y1=0.15)

    # obstacle2=garden_hose.add_obstacle()
    # obstacle2.set_dimensions(x0=0.9, x1=1.2, y0=0.85, y1=1)

    # obstacle3=garden_hose.add_obstacle()
    # obstacle3.set_dimensions(x0=1.2, x1=1.5, y0=0, y1=0.35)

    # obstacle4=garden_hose.add_obstacle()
    # obstacle4.set_dimensions(x0=1.2, x1=1.5, y0=0.65, y1=1)

    # obstacle5=garden_hose.add_obstacle()
    # obstacle5.set_dimensions(x0=1.5, x1=1.8, y0=0, y1=0.45)

    # obstacle6=garden_hose.add_obstacle()
    # obstacle6.set_dimensions(x0=1.5, x1=1.8, y0=0.55, y1=1)

    # F=10
    # garden_hose.set_parameters(dx= 0.02, dy=0.02, dt=10**(-4), mu=0.4, rho=1, U=0, Fx=F)
    # garden_hose.simulate_system(max_iterations=10**5)
    # garden_hose.plot_subplots()
    # print(f"The Reynolds number is equal to: {(garden_hose.Ly*garden_hose.rho*np.average(np.sqrt(garden_hose.u**2+garden_hose.v**2)))/(garden_hose.mu)}")
    return

if __name__=='__main__':
    main()

    
