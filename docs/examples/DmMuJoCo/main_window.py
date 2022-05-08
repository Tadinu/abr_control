import traceback

class MainWindow:
    def __init__(self, in_sim_interface, in_robot_config):
        # Start main window event loop
        self.main_window = None
        self.sim_interface = in_sim_interface
        self.robot_config = in_robot_config

    def exec(self, in_tick):
        try:
            # get the end-effector's initial position
            feedback = self.sim_interface.get_feedback()
            start = self.robot_config.Tx("EE", q=feedback["q"])

            print("\nSimulation starting...\n")  
            print(f'EE start: {start}')  
            self.main_window = self.sim_interface.window
            self.main_window.event_loop(tick_func=in_tick)
        except:
            print(traceback.format_exc())

        finally:
            # close the connection to the arm
            self.sim_interface.disconnect()
            print("Simulation terminated...")
            if self.main_window:
                self.main_window.close()