''' Description:
This Python program simulates a 5-stage pipelined CPU architecture with an 8-bit accumulator 
model. It implements the pipeline stages — Instruction Fetch (IF), Instruction Decode (ID),
Execute (EX), and Memory Access/Write-Back (MEM/WB) — without hazard detection or forwarding.

Sections Overview:
1. Instruction Class Definition:
   Defines the Instruction structure with opcode and operand fields.

2. Processor Architecture State Initialization:
   Declares and initializes CPU registers (ACC, flags Z and N), program counter (PC),
   and both instruction and data memory.

3. Helper Functions:
   - mask8(): Ensures 8-bit arithmetic by masking values.
   - set_flags_from(): Updates zero (Z) and negative (N) flags based on accumulator output.

4. Pipeline Stage Implementations:  opcode (LOAD,STORE.ADD,SUB,AND,OR,LOADI,STOREI,JMP,JN,JZ,HALT)
   - fetch_instruction(): Fetches instructions into IF/ID pipeline register.
   - decode_instruction(): Decodes opcode, extracts operands, and sets control signals.
   - execute_instruction(): Performs arithmetic/logic operations or branch resolution.
   - memory_access(): Handles memory read/write for direct and indirect addressing modes.
   - write_back(): Updates the accumulator and status flags based on results.

5. run function:
   Simulates the pipeline execution cycle-by-cycle, passing data through the pipeline 
   registers, updating state, and generating a trace of CPU activity.

6. Demo Program :
   Demonstrates indirect and direct memory operations using LOADI, STOREI, ADD, AND, LOAD, HALT, JN
   instructions. Memory is preloaded to verify pointer-based behavior.'''


from typing import List, Dict, Any

class Instruction:
    def __init__(self, opcode, operands=None):
        self.opcode = opcode
        self.operands = operands if operands is not None else []

    def __repr__(self):
        return f"Instruction(opcode={self.opcode}, operands={self.operands})"


# Processor Architecture State initilization (Registers, Flags, and Memory)

ACC: int = 0      # 8-bit accumulator reg
Z: int = 0        # zero flag
N: int = 0        # negative flag (bit7)
MEM: Dict[int, int] = {i: 0 for i in range(256)}  # Main Memory
instruction_memory: List[Instruction] = []   # Program Memory
pc: int = 0       # 8-bit Program Counter

# Pipeline registers
pipeline_regs = {
    'IF_ID': None,
    'ID_EX': None,
    'EX_MEM': None,
    'MEM_WB': None
}


# Ensures any value in CPU stays within 8 bits

def mask8(x: int) -> int:
    return x & 0xFF

# updates the status flags
def set_flags_from(value: int):
    """Update Z/N flags from 8-bit value."""
    global Z, N
    v = mask8(value)
    Z = 1 if v == 0 else 0
    N = 1 if (v & 0x80) else 0


# Pipeline Stage implementations (NO hazards/forwarding)

#Instruction Fetch
def fetch_instruction(instruction_memory: List[Instruction], pc_val: int):
    """IF: fetch one instruction; fill IF/ID latch (instr, pc, valid)."""
    if 0 <= pc_val < len(instruction_memory):
        return {
            "instr": instruction_memory[pc_val],  # instr[11:0] (object)
            "pc": mask8(pc_val),                  # pc[7:0]
            "valid": True
        }
    return {"instr": None, "pc": mask8(pc_val), "valid": False}
    
#Instruction Decode
def decode_instruction(IF_ID_reg):
    """ID: classify opcode -> control signals, extract 8-bit operand, snapshot ACC/Z/N."""
    if IF_ID_reg is None or not IF_ID_reg.get("valid", False) or IF_ID_reg["instr"] is None:
        return None

    instr = IF_ID_reg["instr"]
    op = instr.opcode.upper()
    ops = instr.operands

    # Control defaults
    control = {
        "is_jump_uncond": False,
        "is_jump_neg": False,
        "is_jump_zero": False,
        "is_load": False,     # direct: ACC <- MEM[x]
        "is_store": False,    # direct: MEM[x] <- ACC
        "is_loadi": False,    # indirect: ACC <- MEM[ MEM[x] ]
        "is_storei": False,   # indirect: MEM[ MEM[x] ] <- ACC
        "is_alu": False,      # arithmetic with immediate
        "alu_uses_mem": False,
        "is_halt": False,     #Stop execution
        "alu_op": "NOP"
    }
    operand8 = 0

    if op == "NOP":
        pass
    elif op=="HALT":
        control.update({"is_halt": True})

    elif op == "LOADI":
        # LOADI x  => ACC <- MEM[ MEM[x] ]
        control.update({"is_loadi": True})
        operand8 = int(ops[0]) & 0xFF   # & 0xFF — it’s a mask that ensures to keep only the lowest 8 bits of the operand

    elif op == "STOREI":
        # STOREI x => MEM[ MEM[x] ] <- ACC
        control.update({"is_storei": True})
        operand8 = int(ops[0]) & 0xFF

    elif op == "LOAD":
        # LOAD x   => ACC <- MEM[x]
        control.update({"is_load": True})
        operand8 = int(ops[0]) & 0xFF

    elif op == "STORE":
        # STORE x  => MEM[x] <- ACC
        control.update({"is_store": True})
        operand8 = int(ops[0]) & 0xFF

    elif op == "ADD":
        control.update({"is_alu": True, "alu_op": "ADD", "alu_uses_mem": True})
        operand8 = int(ops[0]) & 0xFF

    elif op == "SUB":
        control.update({"is_alu": True, "alu_op": "SUB", "alu_uses_mem": True})
        operand8 = int(ops[0]) & 0xFF
    elif op == "AND":
        control.update({"is_alu": True, "alu_op": "AND", "alu_uses_mem": True})
        operand8 = int(ops[0]) & 0xFF

    elif op == "OR":
        control.update({"is_alu": True, "alu_op": "OR", "alu_uses_mem": True})
        operand8 = int(ops[0]) & 0xFF


    elif op == "JMP":
        control.update({"is_jump_uncond": True})
        operand8 = int(ops[0]) & 0xFF

    elif op == "JZ":
        control.update({"is_jump_zero": True})
        operand8 = int(ops[0]) & 0xFF

    elif op == "JN":
        control.update({"is_jump_neg": True})
        operand8 = int(ops[0]) & 0xFF
        
    else:
        
        op = "NOP"
  #creates the ID/EX pipeline register, which holds the decoded instruction, operand, snapshots of ACC and flags, and all control signals
  #ensuring the next pipeline stage has every piece of information
    return {
        "opcode": op,
        "operand": operand8,                 # N LOADI/STOREI this is the POINTER LOCATION x
        "pc": IF_ID_reg["pc"],
        "acc_snapshot": ACC & 0xFF,
        "flags_snapshot": (Z, N),
        "control": control,
        "valid": True
    }

#Instruction Execution stage
def execute_instruction(ID_EX_reg):
    """
    resolve branches, run ALU for arithmetic, compute base mem_addr.
    Produces EX/MEM latch.
    """
    if ID_EX_reg is None or not ID_EX_reg.get("valid", False):
        return None
    # gets details from ID/EX register
    control = ID_EX_reg["control"]
    operand = ID_EX_reg["operand"]
    acc_val = ID_EX_reg["acc_snapshot"]
    Z_snap, N_snap = ID_EX_reg["flags_snapshot"]

    alu_out = 0
    mem_addr = 0
    
    # Halt -> No Computation, carry it Forward
    if control.get("is_halt", False):
        return {
            "alu_out": 0,
            "operand": 0,
            "acc_snapshot": acc_val,
            "mem_addr": 0,
            "opcode": "HALT",
            "flags_snapshot": (Z_snap, N_snap),
            "control": control,
            "valid": True,
        }
    

    # Control-flow resolution 
    global pc
    if control["is_jump_uncond"]:
        pc = operand
    elif control["is_jump_zero"] and Z_snap == 1:   #set pc = operand if the snapshotted Z flag was 1.
        pc = operand
    elif control["is_jump_neg"] and N_snap == 1:    #set pc = operand if the snapshotted N flag was 1.
        pc = operand

    # ALU path — ONLY for explicit ALU ops (LOADI no longer uses ALU)
    if control["is_alu"] and not control.get("alu_uses_mem", False):
        op = control.get("alu_op", "NOP")
        if op == "ADD":
            alu_out = mask8(acc_val + operand)
        elif op == "SUB":
            alu_out = mask8(acc_val - operand)
        elif op == "AND":
            alu_out = mask8(acc_val & operand)
        elif op == "OR":
            alu_out = mask8(acc_val | operand)
        else:
            alu_out = mask8(acc_val)
        

    # For any memory-class op, carry a base address forward.
    # Carry X forward for LOAD/STORE/LOADI/STOREI and for ALU-with-memory.
    if (control["is_load"] or control["is_store"] or control["is_storei"] or
        control["is_loadi"] or (control["is_alu"] and control.get("alu_uses_mem", False))):
        mem_addr = operand & 0xFF
    
    #This EX/MEM pipeline register. It carries everything MEM needs
    return {
        "alu_out": alu_out,                    # alu_out[7:0] # ALU result for ALU ops (0 otherwise)
        "operand": operand,                    # operand[7:0]
        "acc_snapshot": acc_val,               # acc_snapshot[7:0]
        "mem_addr": mem_addr,                  # mem_addr[7:0] (x for indirect)
        "opcode": ID_EX_reg["opcode"],         # opcode[3:0]
        "flags_snapshot": (Z_snap, N_snap),    # {Z, N}
        "control": control,
        "valid": True
    }

# Instruction Memoery Access
def memory_access(EX_MEM_reg):
    """
    MEM: single-cycle memory.
    Direct loads/stores use mem_addr directly.
    Indirect (LOADI/STOREI) perform a double dereference:
      ptr = MEM[x]; then read/write MEM[ptr].
    Produces MEM/WB latch.
    """
    if EX_MEM_reg is None or not EX_MEM_reg.get("valid", False):
        return None

    control = EX_MEM_reg["control"]
    addr_x = EX_MEM_reg["mem_addr"] & 0xFF   # for LOADI/STOREI this is the pointer location x
    read_val = 0
    alu_out = EX_MEM_reg["alu_out"]  # default (usually 0 here)

    if control["is_load"]:
        # Direct: ACC <- MEM[x]
        read_val = MEM.get(addr_x, 0)

    elif control["is_store"]:
        # Direct: MEM[x] <- ACC
        MEM[addr_x] = EX_MEM_reg["acc_snapshot"] & 0xFF

    elif control["is_loadi"]:
        # Indirect: ACC <- MEM[ MEM[x] ]
        ptr = MEM.get(addr_x, 0) & 0xFF      # first deref: pointer value at x
        read_val = MEM.get(ptr, 0)           # second deref: read target

    elif control["is_storei"]:
        # Indirect: MEM[ MEM[x] ] <- ACC
        ptr = MEM.get(addr_x, 0) & 0xFF      # first deref: pointer value at x
        MEM[ptr] = EX_MEM_reg["acc_snapshot"] & 0xFF  # second: write at target

    # ALU ops that use M[X]:
    elif control["is_alu"] and control.get("alu_uses_mem", False):
        m = MEM.get(addr_x, 0) & 0xFF
        a = EX_MEM_reg["acc_snapshot"] & 0xFF
        op = control.get("alu_op", "NOP")
        if op == "ADD":
            alu_out = mask8(a + m)
        elif op == "SUB":
            alu_out = mask8(a - m)
        elif op == "AND":
            alu_out = mask8(a & m)
        elif op == "OR":
            alu_out = mask8(a | m)

    return {
        "data_mem": read_val,                  # load result (direct or indirect)
        "alu_out": alu_out,
        "opcode": EX_MEM_reg["opcode"],
        "operand": EX_MEM_reg["operand"],
        "flags_snapshot": EX_MEM_reg["flags_snapshot"],
        "control": control,
        "valid": True
    }

#Instruction Writeback
def write_back(MEM_WB_reg, register_file=None):
    """
    WB: 
      - LOAD & LOADI write ACC from data_mem.
      - ALU ops write ACC from alu_out.
      - STORE/STOREI/branches do not write ACC.
    Updates Z/N based on written ACC.
    """
    if MEM_WB_reg is None or not MEM_WB_reg.get("valid", False):
        return

    global ACC
    control = MEM_WB_reg["control"]

    if control["is_load"] or control["is_loadi"]:
        ACC = mask8(MEM_WB_reg["data_mem"])
    elif control["is_alu"] and not (control["is_store"] or control["is_storei"]):
        ACC = mask8(MEM_WB_reg["alu_out"])

    set_flags_from(ACC)


# Simulation Loop

def run(program, max_cycles=20, start_pc=0, initial_mem=None) -> List[Dict[str, Any]]:
    global instruction_memory, pc, ACC, Z, N, MEM, pipeline_regs
    instruction_memory = list(program)
    pc = start_pc & 0xFF
    ACC = 0; Z = 0; N = 0

    # initialize memory with optional preloads
    MEM = {i: 0 for i in range(256)}
    if initial_mem:
        for k, v in initial_mem.items():
            MEM[k & 0xFF] = v & 0xFF
    pipeline_regs = {'IF_ID': None, 'ID_EX': None, 'EX_MEM': None, 'MEM_WB': None}

    trace: List[Dict[str, Any]] = []
    cycles = 0

    def snapshot() -> Dict[str, Any]:
        def dump(reg):
            if reg is None:
                return None
            d = dict(reg)
            if "instr" in d and isinstance(d["instr"], Instruction):
                d["instr"] = {"opcode": d["instr"].opcode, "operands": list(d["instr"].operands)}
            return d
        return {
            "cycle": cycles,
            "pc": pc,
            "ACC": ACC,
            "Z": Z,
            "N": N,
            "IF_ID": dump(pipeline_regs['IF_ID']),
            "ID_EX": dump(pipeline_regs['ID_EX']),
            "EX_MEM": dump(pipeline_regs['EX_MEM']),
            "MEM_WB": dump(pipeline_regs['MEM_WB']),
        }

    while cycles < max_cycles:
        # WB
        write_back(pipeline_regs['MEM_WB'], None)
        # MEM
        pipeline_regs['MEM_WB'] = memory_access(pipeline_regs['EX_MEM'])
        # EX
        pipeline_regs['EX_MEM'] = execute_instruction(pipeline_regs['ID_EX'])
        # ID
        pipeline_regs['ID_EX'] = decode_instruction(pipeline_regs['IF_ID'])
        # IF (uses current pc), then sequential increment 
        pipeline_regs['IF_ID'] = fetch_instruction(instruction_memory, pc)
        pc = (pc + 1) & 0xFF
        
         
        #  Trace helper : prints the useful info
        
        wb = pipeline_regs['MEM_WB']
        if wb and wb.get('valid', False):
            c = wb['control']
            if c.get('is_load') or c.get('is_loadi'):
                print(f"[WB] cycle {cycles}: ACC <- MEM = {wb['data_mem']}")
            elif c.get('is_alu') and not (c.get('is_store') or c.get('is_storei')):
                print(f"[WB] cycle {cycles}: ACC <- ALU = {wb['alu_out']}")

        exm = pipeline_regs['EX_MEM']
        if exm and exm.get('valid', False):
            c = exm['control']
            if c.get('is_store') or c.get('is_storei'):
                if c.get('is_store'):
                    print(f"[MEM] cycle {cycles}: MEM[{exm['mem_addr']}] <- ACC_snapshot={exm['acc_snapshot']}")
                else:
                    print(f"[MEM] cycle {cycles}: STOREI pending: will write ACC_snapshot to MEM[ MEM[{exm['mem_addr']}] ]")
                    
        # ----- HALT detection (stop when HALT reaches WB) 
        if wb and wb.get('valid', False) and wb['control'].get('is_halt', False):
            print(f"[HALT] cycle {cycles}: CPU halted.")
            trace.append(snapshot())
            break


        trace.append(snapshot())
        cycles += 1

    return trace

#------------------------------------------xxxxx-----------------------------------------------------------------------------------------------
# Program to verify implementation

if __name__ == "__main__":
    """
    Program to test:
    LOADI, ADD, STOREI, LOAD, STORE, AND, OR, SUB, JN, HALT

    Memory Setup:
      MEM[5]  = 10     (pointer for LOADI)
      MEM[10] = 50     (value for LOADI’s indirect read and for ADD via M[10])
      MEM[20] = 30     (pointer for STOREI)
      MEM[30] = 0      (target of STOREI; will become 100)
      MEM[25] = 25     (operand used by AND/OR/SUB)
      MEM[40] = 0      (will be written by STORE with 100)
      MEM[60] = 100    (operand for SUB to force negative -> N=1)
      MEM[61] = 1      (operand used on the fallthrough path if JN not taken)
    """

    program = [
        #  0
        Instruction('LOADI', [5]),      # ACC <- M[M[5]] = M[10] = 50
        Instruction('NOP', []),         # 1
        Instruction('NOP', []),         # 2

        Instruction('ADD', [10]),       # 3  ACC <- ACC + M[10] = 50 + 50 = 100
        Instruction('NOP', []),         # 4
        Instruction('NOP', []),         # 5

        Instruction('STOREI', [20]),    # 6  M[M[20]] <- ACC  => M[30] = 100
        Instruction('NOP', []),         # 7
        Instruction('NOP', []),         # 8

        Instruction('STORE', [40]),     # 9  M[40] <- ACC  => 100
        Instruction('NOP', []),         # 10

        Instruction('LOAD', [40]),      # 11 ACC <- M[40] = 100
        Instruction('NOP', []),         # 12
        Instruction('NOP', []),         # 13

        Instruction('AND', [25]),       # 14 ACC <- 100 & 25 = 0
        Instruction('NOP', []),         # 15
        Instruction('NOP', []),         # 16

        Instruction('OR',  [25]),       # 17 ACC <- 0 | 25 = 25
        Instruction('NOP', []),         # 18
        Instruction('NOP', []),         # 19

        Instruction('SUB', [25]),       # 20 ACC <- 25 - 25 = 0  (Z=1, N=0)
        Instruction('NOP', []),         # 21
        Instruction('NOP', []),         # 22

        Instruction('SUB', [60]),       # 23 ACC <- 0 - 100 = 156 (0x9C), N=1
        Instruction('NOP', []),         # 24
        Instruction('NOP', []),         # 25

        Instruction('JN',  [32]),       # 26 if N=1 jump to index 32 (target below)
        Instruction('NOP', []),         # 27 padding so prefetched instrs are harmless
        Instruction('NOP', []),         # 28

        # Fallthrough path (should be skipped because N=1)
        Instruction('ADD', [61]),       # 29 would do ACC <- ACC + 1
        Instruction('STORE', [50]),     # 30 would store to M[50]
        Instruction('HALT', []),        # 31 would halt if we didn't jump

        # Jump target:
        Instruction('LOAD', [30]),      # 32 ACC <- M[30] = 100  (stored earlier by STOREI)
        Instruction('NOP', []),         # 33 padding to let LOAD reach WB before HALT
        Instruction('HALT', []),        # 34 final stop
    ]

    initial_mem = {
        5: 10,
        10: 50,
        20: 30,
        30: 0,
        25: 25,
        40: 0,
        60: 100,
        61: 1,
    }

    t = run(program, max_cycles=120, start_pc=0, initial_mem=initial_mem)

    # show early cycles
    for row in t[:14]:
        print(row)

    # Final architectural state
    last = t[-1]
    print("\nFINAL STATE")
    print("ACC:", last["ACC"], "Z:", last["Z"], "N:", last["N"])
    print(f"MEM[10]: {MEM[10]}, MEM[20]: {MEM[20]}, MEM[30]: {MEM[30]}, MEM[40]: {MEM[40]}, MEM[50]: {MEM.get(50,0)}")
    print(f"Total cycles executed: {len(t)}")

    # Simple checks 
    assert last["ACC"] == 100 and last["Z"] == 0 and last["N"] == 0, "Final ACC/Z/N mismatch"
    assert MEM[30] == 100, "MEM[30] should be 100 (written by STOREI)"
    assert MEM[40] == 100, "MEM[40] should be 100 (written by STORE)"
    assert MEM.get(50, 0) == 0, "MEM[50] should be 0 (fallthrough skipped by JN)"
    print(" All expected results match.")