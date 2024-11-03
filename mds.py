class MotivationalDriveSystem:
    def __init__(self):
        # Dictionary to store drive levels
        self.drives = {
            'exploration': 1.0,  # Drive to explore new data
            'learning': 1.0,     # Drive for knowledge acquisition
            'social_interaction': 1.0  # Drive to engage with external entities
        }
        # Thresholds for drives, below which motivation drops
        self.drive_thresholds = {
            'exploration': 0.5,
            'learning': 0.5,
            'social_interaction': 0.5
        }

    def modulate_drive(self, drive_name, change):
        # Modify the level of a given drive
        if drive_name in self.drives:
            self.drives[drive_name] = max(0.0, min(2.0, self.drives[drive_name] + change))

    def check_motivation(self):
        # Evaluate which drives are active and return their states
        active_drives = {drive: level for drive, level in self.drives.items() if level > self.drive_thresholds[drive]}
        return active_drives

    def display_drive_states(self):
        # Display the current drive levels
        return {drive: round(level, 2) for drive, level in self.drives.items()}

# Integrate MDS with Advanced SNS
class CompleteCognitiveArchitecture(AdvancedSyntheticNeuralSubstrate):
    def __init__(self, input_size, hidden_size, output_size):
        super(CompleteCognitiveArchitecture, self).__init__(input_size, hidden_size, output_size)
        self.mds = MotivationalDriveSystem()  # Initialize the MDS

    def adjust_drives(self, drive_name, change):
        # Adjust the MDS drive levels
        self.mds.modulate_drive(drive_name, change)

    def evaluate_motivation(self):
        # Evaluate current active drives
        return self.mds.check_motivation()

    def get_drive_states(self):
        # Display the current drive states
        return self.mds.display_drive_states()

# Instantiate the complete cognitive architecture with MDS
input_size = 10  # Number of input features
hidden_size = 50  # Number of neurons in hidden layers
output_size = 5  # Number of output features

cognitive_architecture = CompleteCognitiveArchitecture(input_size, hidden_size, output_size)

# Adjust drive levels and test motivation evaluation
cognitive_architecture.adjust_drives('exploration', 0.3)  # Increase exploration drive
cognitive_architecture.adjust_drives('learning', -0.2)   # Decrease learning drive

# Print active drives and current drive states
print("Active Drives:", cognitive_architecture.evaluate_motivation())
print("Drive States:", cognitive_architecture.get_drive_states())

# Run a sample input and observe how MDS affects the process
input_tensor = torch.randn(1, input_size)
output = cognitive_architecture(input_tensor, context="Motivation evaluation test")
print("Output:", output)
