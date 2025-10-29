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
from process_mem import Instruction

# Processor Architecture State initilization (Registers, Flags, and Memory)

ACC: int = 0      # 8-bit accumulator reg
Z: int = 0        # zero flag
N: int = 0        # negative flag (bit7)
MEM: Dict[int, int] = {i: 0 for i in range(256)}  # Main Memory
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
def fetch_instruction(memory: Dict[int, int], pc_val: int):
    """IF: fetch one instruction; fill IF/ID latch (instr, pc, valid)."""
    if 0 <= pc_val < 0xFF:
        instruction = Instruction.binary_to_instruction(memory[pc_val])
        return {
            "instr": instruction,  # instr[11:0] (object)
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
    op = instr.opcode_to_string().upper()

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
        "alu_op": 14,          # NOP by default
    }
    operand8 = 0

    if op == "NOP":
        pass
    elif op=="HALT":
        control.update({"is_halt": True})

    elif op == "LOADI":
        # LOADI x  => ACC <- MEM[ MEM[x] ]
        control.update({"is_loadi": True})
        operand8 = int(instr.operand) & 0xFF   # & 0xFF — it’s a mask that ensures to keep only the lowest 8 bits of the operand

    elif op == "STOREI":
        # STOREI x => MEM[ MEM[x] ] <- ACC
        control.update({"is_storei": True})
        operand8 = int(instr.operand) & 0xFF

    elif op == "LOAD":
        # LOAD x   => ACC <- MEM[x]
        control.update({"is_load": True})
        operand8 = int(instr.operand) & 0xFF

    elif op == "STORE":
        # STORE x  => MEM[x] <- ACC
        control.update({"is_store": True})
        operand8 = int(instr.operand) & 0xFF

    elif op == "ADD":
        control.update({"is_alu": True, "alu_op": Instruction.opcode_to_int("ADD"), "alu_uses_mem": True})
        operand8 = int(instr.operand) & 0xFF

    elif op == "SUB":
        control.update({"is_alu": True, "alu_op": Instruction.opcode_to_int("SUB"), "alu_uses_mem": True})
        operand8 = int(instr.operand) & 0xFF
    elif op == "AND":
        control.update({"is_alu": True, "alu_op": Instruction.opcode_to_int("AND"), "alu_uses_mem": True})
        operand8 = int(instr.operand) & 0xFF

    elif op == "OR":
        control.update({"is_alu": True, "alu_op": Instruction.opcode_to_int("OR"), "alu_uses_mem": True})
        operand8 = int(instr.operand) & 0xFF

    elif op == "JMP":
        control.update({"is_jump_uncond": True})
        operand8 = int(instr.operand) & 0xFF

    elif op == "JZ":
        control.update({"is_jump_zero": True})
        operand8 = int(instr.operand) & 0xFF

    elif op == "JN":
        control.update({"is_jump_neg": True})
        operand8 = int(instr.operand) & 0xFF
      
    else:      
        op = "NOP"
    #creates the ID/EX pipeline register, which holds the decoded instruction, operand, snapshots of ACC and flags, and all control signals
    #ensuring the next pipeline stage has every piece of information
    return {
        "opcode": instr.opcode,
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

    acc_val = forwarding(
    ID_EX_reg,
    pipeline_regs.get('EX_MEM'),
    pipeline_regs.get('MEM_WB'),
    ID_EX_reg["acc_snapshot"]
    )
    
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
            "opcode": Instruction.opcode_to_int("HALT"),  # HALT opcode
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
        if op == Instruction.opcode_to_int("ADD"):
            alu_out = mask8(acc_val + operand)
        elif op == Instruction.opcode_to_int("SUB"):
            alu_out = mask8(acc_val - operand)
        elif op == Instruction.opcode_to_int("AND"):
            alu_out = mask8(acc_val & operand)
        elif op == Instruction.opcode_to_int("OR"):
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
        if op == Instruction.opcode_to_int("ADD"):
            alu_out = mask8(a + m)
        elif op == Instruction.opcode_to_int("SUB"):
            alu_out = mask8(a - m)
        elif op == Instruction.opcode_to_int("AND"):
            alu_out = mask8(a & m)
        elif op == Instruction.opcode_to_int("OR"):
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

# Hazard Detection
def hazard_detection(IF_ID_reg, ID_EX_reg, EX_MEM_reg):
    """Detect data and control hazards."""
    stall = False
    flush = False

    # Load-use hazard
    if ID_EX_reg and ID_EX_reg.get("valid", False):
        c = ID_EX_reg["control"]
        if c.get("is_load") or c.get("is_loadi"):
            if IF_ID_reg and IF_ID_reg.get("valid", False):
                stall = True

    # Control hazard (jump/branch taken)
    if EX_MEM_reg and EX_MEM_reg.get("valid", False):
        ctrl = EX_MEM_reg["control"]
        fZ, fN = EX_MEM_reg["flags_snapshot"]
        if (ctrl.get("is_jump_uncond") or
            (ctrl.get("is_jump_zero") and fZ == 1) or
            (ctrl.get("is_jump_neg") and fN == 1)):
            flush = True

    return stall, flush

#Data forwording
def forwarding(ID_EX_reg, EX_MEM_reg, MEM_WB_reg, acc_snapshot):
    """Return accumulator value with forwarding applied (for ALU input)."""
    acc_val = acc_snapshot

    # Forward from EX/MEM if available
    if EX_MEM_reg and EX_MEM_reg.get("valid", False):
        c = EX_MEM_reg["control"]
        if c.get("is_alu"):
            acc_val = EX_MEM_reg["alu_out"]

    # Forward from MEM/WB (highest priority)
    if MEM_WB_reg and MEM_WB_reg.get("valid", False):
        c = MEM_WB_reg["control"]
        if c.get("is_alu") or c.get("is_load") or c.get("is_loadi"):
            acc_val = MEM_WB_reg.get("data_mem", MEM_WB_reg.get("alu_out", acc_val))

    return acc_val

# Simulation Loop

def run(initial_mem, max_cycles=20, start_pc=0) -> List[Dict[str, Any]]:
    global MEM, pc, ACC, Z, N, MEM, pipeline_regs
    pc = start_pc & 0xFF
    ACC = 0; Z = 0; N = 0

    # initialize memory
    MEM = {i: 0 for i in range(256)}
    if initial_mem:
        for k, v in initial_mem.items():
            MEM[k & 0xFF] = v & 0xFFF
    pipeline_regs = {'IF_ID': None, 'ID_EX': None, 'EX_MEM': None, 'MEM_WB': None}

    trace: List[Dict[str, Any]] = []
    cycles = 0

    def snapshot() -> Dict[str, Any]:
        def dump(reg):
            if reg is None:
                return None
            d = dict(reg)
            if "instr" in d and isinstance(d["instr"], Instruction):
                d["instr"] = {"opcode": d["instr"].opcode, "operand": d["instr"].operand}

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
        # HAZARD DETECTION
        stall, flush = hazard_detection(
            pipeline_regs['IF_ID'],
            pipeline_regs['ID_EX'],
            pipeline_regs['EX_MEM']
        )

        if flush:
            print(f"[CYCLE {cycles}] Control hazard = Flushing IF/ID and ID/EX")
            pipeline_regs['IF_ID'] = None
            pipeline_regs['ID_EX'] = None

        if stall:
            print(f"[CYCLE {cycles}] Data hazard = Stalling pipeline 1 cycle")
            pc = (pc - 1) & 0xFF  # freeze fetch
            pipeline_regs['ID_EX'] = None
        # ID
        pipeline_regs['ID_EX'] = decode_instruction(pipeline_regs['IF_ID'])
        # IF (uses current pc), then sequential increment 
        pipeline_regs['IF_ID'] = fetch_instruction(MEM, pc)
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

#-----------------------------------------------------------------------------------------------------------------------------------------
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
        Instruction('LOADI', 40),      # ACC <- M[M[5]] = M[10] = 50
        Instruction('NOP'),           # 1
        Instruction('NOP'),           # 2

        Instruction('ADD', 41),       # 3  ACC <- ACC + M[10] = 50 + 50 = 100
        Instruction('NOP'),           # 4
        Instruction('NOP'),           # 5

        Instruction('STOREI', 42),    # 6  M[M[20]] <- ACC  => M[30] = 100
        Instruction('NOP'),           # 7
        Instruction('NOP'),           # 8

        Instruction('STORE', 45),     # 9  M[40] <- ACC  => 100
        Instruction('NOP'),           # 10

        Instruction('LOAD', 45),      # 11 ACC <- M[40] = 100
        Instruction('NOP'),           # 12
        Instruction('NOP'),           # 13

        Instruction('AND', 44),       # 14 ACC <- 100 & 25 = 0
        Instruction('NOP'),           # 15
        Instruction('NOP'),           # 16

        Instruction('OR', 44),        # 17 ACC <- 0 | 25 = 25
        Instruction('NOP'),           # 18
        Instruction('NOP'),           # 19

        Instruction('SUB', 44),       # 20 ACC <- 25 - 25 = 0  (Z=1, N=0)
        Instruction('NOP'),           # 21
        Instruction('NOP'),           # 22

        Instruction('SUB', 60),       # 23 ACC <- 0 - 100 = 156 (0x9C), N=1
        Instruction('NOP'),           # 24
        Instruction('NOP'),           # 25

        Instruction('JN', 32),        # 26 if N=1 jump to index 32 (target below)
        Instruction('NOP'),           # 27 padding so prefetched instrs are harmless
        Instruction('NOP'),           # 28

        # Fallthrough path (should be skipped because N=1)
        Instruction('ADD', 61),       # 29 would do ACC <- ACC + 1
        Instruction('STORE', 50),     # 30 would store to M[50]
        Instruction('HALT'),          # 31 would halt if we didn't jump

        # Jump target:
        Instruction('LOAD', 43),      # 32 ACC <- M[30] = 100  (stored earlier by STOREI)
        Instruction('NOP'),           # 33 padding to let LOAD reach WB before HALT
        Instruction('HALT'),          # 34 final stop
    ]

    preloaded_mem = {
        40: 41,
        41: 50,
        42: 43,
        43: 0,
        44: 25,
        45: 0,
        60: 100,
        61: 1,
    }

    initial_mem = {}
    for i, instr in enumerate(program):
        initial_mem[i] = instr.instruction_to_binary()
    for k, v in preloaded_mem.items():
        initial_mem[k] = v & 0xFF
    print("Initial Memory: ")
    for loc in sorted(initial_mem.keys()):
        val = initial_mem[loc]
        Instruction_inst = Instruction.binary_to_instruction(val)
        print(f"MEM[{loc:02}] = {Instruction_inst.opcode_to_string()} {Instruction_inst.operand}")

    t = run(max_cycles=120, start_pc=0, initial_mem=initial_mem)

    # show early cycles
    for row in t[:14]:
        print(row)

    # Final architectural state
    last = t[-1]
    print("\nFINAL STATE")
    print("ACC:", last["ACC"], "Z:", last["Z"], "N:", last["N"])
    print(f"MEM[41]: {MEM[41]}, MEM[42]: {MEM[42]}, MEM[43]: {MEM[43]}, MEM[44]: {MEM[44]}, MEM[45]: {MEM.get(45,0)}")
    print(f"Total cycles executed: {len(t)}")

    # Simple checks 
    assert last["ACC"] == 100 and last["Z"] == 0 and last["N"] == 0, "Final ACC/Z/N mismatch"
    assert MEM[43] == 100, "MEM[43] should be 100 (written by STOREI)"
    assert MEM[45] == 100, "MEM[40] should be 100 (written by STORE)"
    assert MEM.get(50, 0) == 0, "MEM[50] should be 0 (fallthrough skipped by JN)"
    print(" All expected results match.")
