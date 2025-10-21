class Instruction:
    def __init__(self, opcode, operands=None):
        self.opcode = opcode
        self.operands = operands if operands is not None else []

    def __repr__(self):
        return f"Instruction(opcode={self.opcode}, operands={self.operands})"
    
pipeline_regs = {
        'IF_ID': None,
        'ID_EX': None,
        'EX_MEM': None,
        'MEM_WB': None
}

def fetch_instruction(instruction_memory, pc):
    if pc < len(instruction_memory):
        return instruction_memory[pc]
    return None

def decode_instruction(instruction):
    if instruction is None:
        return None

def execute_instruction(instruction):
    if instruction is None:
        return None

def memory_access(instruction):
    if instruction is None:
        return None


def write_back(instruction, register_file):
    if instruction is None:
        return
    
def main():
    instruction_memory = [
        Instruction('LOAD', ['R1', '0(R2)']),
        Instruction('ADD', ['R3', 'R1', 'R4']),
        Instruction('STORE', ['R3', '0(R5)']),
    ]
    
    register_file = {f'R{i}': 0 for i in range(32)}
    pc = 0
    cycles = 0
    max_cycles = 10

    while cycles < max_cycles:
        # Write Back
        write_back(pipeline_regs['MEM_WB'], register_file)
        
        # Memory Access
        pipeline_regs['MEM_WB'] = memory_access(pipeline_regs['EX_MEM'])
        
        # Execute
        pipeline_regs['EX_MEM'] = execute_instruction(pipeline_regs['ID_EX'])
        
        # Decode
        pipeline_regs['ID_EX'] = decode_instruction(pipeline_regs['IF_ID'])
        
        # Fetch
        pipeline_regs['IF_ID'] = fetch_instruction(instruction_memory, pc)
        pc += 1
        
        cycles += 1
    