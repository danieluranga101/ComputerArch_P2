class S12PipelineSimulator:
    def __init__(self, instruction_memory):
        self.instruction_memory = instruction_memory
        self.data_memory = [0] * 256  # Example data memory size
        self.pc = 0
        self.accumulator = 0
        self.halted = False
        self.pipeline_regs = {
            'IF_ID': [None, False],  # [instruction, valid]
            'ID_EX': [None, False],
            'EX_MEM': [None, False],
            'MEM_WB': [None, False],
        }
        self.z_flag = False
        self.n_flag = False
        self.stall = False


    class Instruction:
        def __init__(self, opcode, operands=None):
            self.opcode = opcode
            self.operands = operands if operands is not None else []

        def __repr__(self):
            return f"Instruction(opcode={self.opcode}, operands={self.operands})"

    def fetch_instruction(self):
        if self.pc < len(self.instruction_memory):
            return self.instruction_memory[self.pc]
        return None

    def decode_instruction(self, instruction):
        if instruction is None:
            return None
            
        # In decode stage, we prepare the operands and control signals
        return instruction

    def execute_instruction(self, instruction):
        if instruction is None:
            return None
            
        result = None
        return result

    def memory_access(self, instruction):
        if instruction is None:
            return None
            
        return instruction

    def write_back(self, instruction):
        if instruction is None:
            return
        return instruction
        
    def simulate_cycle(self):
        if self.halted:
            return
        # Fetch
        self.fetch_instruction()
        # Decode
        self.decode_instruction(self.pipeline_regs['IF_ID'])
        # Execute
        self.execute_instruction(self.pipeline_regs['ID_EX'])
        # Write Back
        self.write_back(self.pipeline_regs['MEM_WB'])
        # Memory Access
        self.memory_access(self.pipeline_regs['EX_MEM'])

def main():
    instruction_memory = [
        S12PipelineSimulator.Instruction('ADD', ['R1', 'R2']),
        S12PipelineSimulator.Instruction('SUB', ['R3', 'R4']),
        # Add more instructions as needed
    ]
    
    simulator = S12PipelineSimulator(instruction_memory)
    
    cycles = 10
    for _ in range(cycles):
        simulator.simulate_cycle()
