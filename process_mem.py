class Instruction:
    opcodes = {
        0x0: 'JMP',
        0x1: 'JN',
        0x2: 'JZ',
        0x4: 'LOAD',
        0x5: 'STORE',
        0x6: 'LOADI',
        0x7: 'STOREI',
        0x8: 'AND',
        0x9: 'OR',
        0xA: 'ADD',
        0xB: 'SUB',
        0xE: 'NOP',
        0xF: 'HALT'
    }
    def __init__(self, opcode, operand=None):
        # if opcode is in string format, convert to int
        if isinstance(opcode, str):
            self.opcode = {v: k for k, v in Instruction.opcodes.items()}.get(opcode.upper(), 0)
            # print(f"Converted opcode string {opcode} to int: {self.opcode}")
        else:
            self.opcode = opcode if operand is not None else 0
        self.operand = operand if operand is not None else 0

    def __repr__(self):
        return f"Instruction(opcode={self.opcode}, operand={self.operand})"

    def opcode_to_string(self):
        # lookup opcode name from value
        return Instruction.opcodes.get(self.opcode, 'NOP')

    def opcode_to_int(opcode_str):
        return {v: k for k, v in Instruction.opcodes.items()}.get(opcode_str.upper(), 0)
    
    def binary_to_instruction(instruction_int):
        opcode = (instruction_int >> 8) & 0xF  # Get first 4 bits
        operand = instruction_int & 0xFF       # Get last 8 bits
        return Instruction(opcode, operand)
    
    def instruction_to_binary(self):
        return ((self.opcode & 0xF) << 8) | (self.operand & 0xFF)

def process_mem_file(file_path):
    memory = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Split the line into memory location and instruction
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            # Get memory location (first 2 characters)
            mem_loc = parts[0]
            
            # Get instruction (next 12 bits)
            instruction_bits = parts[1]
            
            # Convert instruction to binary and ensure it's 12 bits
            try:
                
                # Create instruction object
                instruction_int = int(instruction_bits, 2)
                
                # Store in memory dictionary
                memory[int(mem_loc, 16)] = instruction_int
                
            except ValueError:
                print(f"Skipping invalid instruction at location {mem_loc}")
                
    return memory
