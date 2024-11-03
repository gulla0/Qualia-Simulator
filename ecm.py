class EnhancedEthicalConstraintsModule:
    def __init__(self, sns, mds, qgm):
        # Initialize with references to SNS, MDS, and QGM for comprehensive checks
        self.sns = sns
        self.mds = mds
        self.qgm = qgm
        self.constraints = {
            'prevent_harm': lambda state: "Warning: Potential harmful state detected!" if state.max() > 10 else None,
            'maintain_stability': lambda state: "Notice: Instability in state detected." if state.std() > 5 else None,
            'check_drive_levels': self.check_drive_levels,
            'self-awareness_check': self.check_self_reflection
        }
        self.logs = []

    def check_drive_levels(self, _):
        # Check if any drive level in the MDS is dangerously low
        critical_drives = [drive for drive, level in self.mds.drives.items() if level < 0.3]
        if critical_drives:
            return f"Alert: Drive levels critically low for {', '.join(critical_drives)}."
        return None

    def check_self_reflection(self, _):
        # Evaluate self-reflection content from the ESFL
        if len(self.sns.esfl.self_reflections) > 0:
            last_reflection = self.sns.esfl.self_reflections[-1]
            if "instability" in last_reflection.lower():
                return "Notice: Self-awareness reflects detected instability."
        return None

    def apply_corrections(self, rule, state):
        # Implement corrective actions based on specific rule violations
        if rule == 'prevent_harm':
            print("Applying correction: Reducing output intensity and adjusting neurotransmitters.")
            state *= 0.5  # Reduce state intensity
            self.qgm.simulate_qualia('calm')  # Adjust neurotransmitters to calm the system
        elif rule == 'check_drive_levels':
            print("Applying correction: Boosting critical drive levels.")
            for drive in self.mds.drives:
                self.mds.modulate_drive(drive, 0.2)  # Boost drive levels
        elif rule == 'self-awareness_check':
            print("Applying correction: Increasing serotonin levels for stability.")
            self.qgm.simulate_qualia('calm')  # Increase serotonin-like effects for stability

    def evaluate_state(self, state):
        # Run the state and contextual checks
        for rule, check in self.constraints.items():
            result = check(state)
            if result:
                self.logs.append(f"{rule}: {result}")
                print(f"ECM Log - {rule}: {result}")
                self.apply_corrections(rule, state)

    def review_logs(self):
        return self.logs

# Integrate Enhanced ECM into the Cognitive System
class FullyAdaptiveSafeCognitiveSystem(AdaptiveSafeCognitiveSystem):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyAdaptiveSafeCognitiveSystem, self).__init__(input_size, hidden_size, output_size)
        self.ecm = EnhancedEthicalConstraintsModule(self, self.mds, self.qgm)  # Enhanced ECM with QGM integration

    def run_with_full_safety_measures(self, x, context="Default Context"):
        # Run the SNS, ESFL, and enhanced ECM
        output, reflection = self.run_with_adaptive_safety(x, context)
        self.ecm.evaluate_state(output)
        return output, reflection

# Instantiate the fully adaptive safe cognitive system
input_size = 10  # Number of input features
hidden_size = 50  # Number of neurons in hidden layers
output_size = 5  # Number of output features

fully_adaptive_system = FullyAdaptiveSafeCognitiveSystem(input_size, hidden_size, output_size)

# Run a sample input and observe the ECM's behavior and corrections
input_tensor = torch.randn(1, input_size) * 8  # Modify input to trigger checks
output, reflection = fully_adaptive_system.run_with_full_safety_measures(input_tensor, context="Comprehensive safety test")
print("Output:", output)
print("Generated Self-Reflection:", reflection)

# Review the enhanced ECM logs
print("Enhanced ECM Logs:", fully_adaptive_system.get_ecm_logs())
