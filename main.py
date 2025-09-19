from navierstokes import *

def main():
    input('Let\'s showcase the navierstokes package! Press enter to continue.\n')
    # driven_cavity()
    # laminar_flow()
    # obstacle()
    trough()
    return

def driven_cavity():
    print("DRIVEN CAVITY")

    input('For our first demonstration, we will create the driven cavity. First, we will initialise a NewtonianFluid object with Lx=2, Ly=2, h=0.03, and dt=0.001. Press enter to continue.\n')
    driven_cavity=NewtonianFluid(2, 2, 0.03, 0.03, 10**-4)

    input('Next, we will set the parameters for this simulation using set_parameters(). We will leave rho as its default value, and set mu=0.8 and U=10. From this, our program will calculate the Reynolds number. Press enter to set the parameters.')
    mu=0.8
    U=5
    driven_cavity.set_parameters(mu=mu, U=U)

    input('Before we go any further, it is wise to check the convergence criterium using check_stability(). Press enter to check the stability.')
    driven_cavity.check_stability()

    input('Now, we only have to set the correct boundary conditions for the system. By default, all walls are no-slip, so we only have to change the BVC for the upper bound. Press enter to set boundary conditions.\n')
    driven_cavity.set_boundary_conditions_p(BVC_up=0)
    driven_cavity.set_boundary_conditions_u(BVC_up=5)

    input('Finally, we can simulate our system. By default, the tolerance is set to 10^-4, and the iterations cap is set at 10^4. Press enter to simulate.')
    driven_cavity.simulate_system()

    input("Now, we have a variety of different plots to show the system. A good method to give a general idea of the system is plot_subplots(). Press enter to plot the system.\n")
    driven_cavity.plot_subplots()

    input("Another interesting method is the plot_profiles method, which plots horizontal and vertical speed profiles of the system at given cross sections. Press enter to plot the speed profiles. \n")
    driven_cavity.plot_profiles(x=0.2, y=0.1)
    return

def laminar_flow():
    print("LAMINAR FLOW")
    print("Next, we will simulate laminar flow through a pipe. I will choose the parameters Lx=2, Ly=1, h=0.03, dt=10^-4, rho=1, mu=1. Now, we will also have a force term, whech we will set to Fx=10.")
    input("However, in contrast to the driven cavity, we cannot yet compute the reynolds number, as we do not know the typical speed of the system until after we've simulated the system. Because of this, the code will display the Reynolds number as zero. Press enter to set the parameters")
    laminar_flow=NewtonianFluid(Lx=2, Ly=1, dx=0.03, dy=0.03, dt=10**-4)

    mu=1
    Fx=10
    laminar_flow.set_parameters(mu=mu, Fx=Fx)

    input("Press enter to check the stability.")
    laminar_flow.check_stability()

    input("Next, we will set the left and right boundary conditions to periodic BVC. Press enter to set boundary conditions. \n")
    laminar_flow.set_boundary_conditions_p(BVC_left="Periodic", BVC_right="Periodic")
    laminar_flow.set_boundary_conditions_u(BVC_left="Periodic", BVC_right="Periodic")
    laminar_flow.set_boundary_conditions_v(BVC_left="Periodic", BVC_right="Periodic")

    input("Press enter to simulate the system.")
    laminar_flow.simulate_system()

    input("Now, we can calculate the Reynolds number by using the update_reynolds_number() method. Press enter to update Reynolds number.")
    laminar_flow.update_reynolds_number()

    input("Press enter to plot a streamline plot using plot_streamline(). The line thickness is proportional to the norm of the speed. \n")
    laminar_flow.plot_streamline()

    input("Press enter to plot a horizontal speed profile.")
    print("We can see that the speed profile is quadratic as expected.\n")
    laminar_flow.plot_u_profile()

    return

def obstacle():
    print("OBSTACLE")
    input("Next, we will simulate laminar flow over an obstacle. The parameters will be: Lx=3, Ly=1, h=0.03, dt=10^-4, rho=1, mu=0.2, Fx=50. Press enter to initialise NewtonialFluid object.")
    obstacle_flow=NewtonianFluid(Lx=3, Ly=1, dx=0.03, dy=0.03, dt=10**-4)

    mu=0.2
    Fx=50
    obstacle_flow.set_parameters(mu=mu, Fx=Fx)
    obstacle_flow.set_boundary_conditions_p(BVC_left="Periodic", BVC_right="Periodic")
    obstacle_flow.set_boundary_conditions_u(BVC_left="Periodic", BVC_right="Periodic")
    obstacle_flow.set_boundary_conditions_v(BVC_left="Periodic", BVC_right="Periodic")

    print("Next, we will add an object in the middle of the pipe using the add_obstacle() method.")
    input("Then, we will use the set_dimensions() method to define the bound of the object. By default, the obstacle is no-slip, so we will not change the BVC. Press enter to create an obtacle. \n")
    obstacle=obstacle_flow.add_obstacle()
    obstacle.set_dimensions(x0=1.3, x1=1.7, y0=0, y1=0.4)


    input("Press enter to simulate system and update the Reynolds number.")
    obstacle_flow.simulate_system(max_iterations=10**5)
    obstacle_flow.update_reynolds_number()

    input("Press enter to plot subplots. \n")
    obstacle_flow.plot_subplots()

def trough():
    print("TROUGH")
    input("For out final demo, we will simulate flow over a trough. The parameters will be almost exactly the same as the last simulation, but with Fx=200. Press enter to initialize system.")
    trough_flow=NewtonianFluid(Lx=3, Ly=1, dx=0.03, dy=0.03, dt=10**-4)

    mu=0.2
    Fx=200
    trough_flow.set_parameters(mu=mu, Fx=Fx)
    trough_flow.set_boundary_conditions_p(BVC_left="Periodic", BVC_right="Periodic")
    trough_flow.set_boundary_conditions_u(BVC_left="Periodic", BVC_right="Periodic")
    trough_flow.set_boundary_conditions_v(BVC_left="Periodic", BVC_right="Periodic")

    input("This time, we will add two obstacles to simulate a trough. Press enter to add obstacles. \n")
    obstacle1=trough_flow.add_obstacle()
    obstacle1.set_dimensions(x0=0, x1=1, y0=0, y1=0.4)
    obstacle2=trough_flow.add_obstacle()
    obstacle2.set_dimensions(x0=2, x1=3, y0=0, y1=0.4)

    input("Press enter to simulate system and update the Reynolds number.")
    trough_flow.simulate_system(max_iterations=10**5)
    trough_flow.update_reynolds_number()

    input("Press enter to plot subplots. \n")
    trough_flow.plot_subplots()

if __name__=='__main__':
    main()