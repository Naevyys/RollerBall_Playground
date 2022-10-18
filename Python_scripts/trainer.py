from experience import Experience, Trajectory, Buffer


class Trainer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_trajectories(self):  # Generates data to train with
        raise NotImplementedError
    
    @staticmethod
    def update_q_network(self):  # Training happens here
        raise NotImplementedError