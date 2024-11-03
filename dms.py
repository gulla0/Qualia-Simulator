class DynamicMemorySystem:
    def __init__(self):
        # Memory structure to store both episodic and semantic memories
        self.episodic_memory = []  # List of dictionaries representing past events
        self.semantic_memory = {}  # Dictionary for storing knowledge and learned concepts

    def store_experience(self, state, context):
        # Store an episodic memory with the current state and associated context
        memory_entry = {
            'state': state.clone().detach().tolist(),
            'context': context  # Context could be a string description or metadata
        }
        self.episodic_memory.append(memory_entry)

    def store_knowledge(self, concept, data):
        # Store knowledge in the semantic memory
        self.semantic_memory[concept] = data

    def retrieve_recent_experience(self, n=1):
        # Retrieve the most recent 'n' experiences
        return self.episodic_memory[-n:] if self.episodic_memory else []

    def retrieve_knowledge(self, concept):
        # Retrieve a concept from semantic memory
        return self.semantic_memory.get(concept, "Concept not found")

# Integration of DMS with Enhanced SNS
class AdvancedSyntheticNeuralSubstrate(EnhancedSyntheticNeuralSubstrate):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdvancedSyntheticNeuralSubstrate, self).__init__(input_size, hidden_size, output_size)
        self.dms = DynamicMemorySystem()  # Initialize the DMS

    def forward(self, x, context="Default Context"):
        # Perform the forward pass, update the RSM, and store the experience in DMS
        output = super().forward(x)
        self.dms.store_experience(output, context)
        return output

    def review_episodic_memory(self):
        # Review the episodic memory stored in the DMS
        return self.dms.episodic_memory

    def add_semantic_knowledge(self, concept, data):
        # Store knowledge in the DMS's semantic memory
        self.dms.store_knowledge(concept, data)

    def retrieve_semantic_knowledge(self, concept):
        # Retrieve a concept from the semantic memory
        return self.dms.retrieve_knowledge(concept)

# Instantiate the advanced SNS with DMS
input_size = 10  # Number of input features
hidden_size = 50  # Number of neurons in hidden layers
output_size = 5  # Number of output features

advanced_sns = AdvancedSyntheticNeuralSubstrate(input_size, hidden_size, output_size)

# Test the model by running sample input with a context
input_tensor = torch.randn(1, input_size)
output = advanced_sns(input_tensor, context="Testing memory storage")

# Print the output and review the episodic memory stored by the DMS
print("Output:", output)
print("Episodic Memory Review:", advanced_sns.review_episodic_memory())

# Add and retrieve semantic knowledge
advanced_sns.add_semantic_knowledge("Neural Network Basics", "A set of algorithms modeled after the human brain")
print("Semantic Knowledge Retrieval:", advanced_sns.retrieve_semantic_knowledge("Neural Network Basics"))
