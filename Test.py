"""
S12 Pipelined CPU Simulator (Universal .mem Loader)
----------------------------------------------------
This simulator can run any .mem file produced for the S12 ISA.

Features:
 - 5-stage pipeline (IF, ID, EX, MEM, WB)
 - Data hazard detection and forwarding
 - Control hazard flush handling
 - Universal .mem file support (automatic binary decode)
 - Compatible with instruction encodings from S12 assembler

Usage:
> python s12_pipeline_with_Hazard_forwarding.py <file.mem>
"""

import sys
from typing import List, Dict, Any

import sys
if len(sys.argv) < 2:
        print("Usage: python test.py <file.mem>")
        sys.exit(1)
# =========================
# Global Stats and Counters
# =========================
instruction_trace = []
instruction_mix = {}
stall_count = 0
forwarding_count = 0

# ===========================
# Core Architecture Components
# ===========================

class Instruction:
    def __init__(self, opcode, operand=None):
        self.opcode = opcode
        self.operand = operand if operand is not None else 0

    def __repr__(self):
        return f"Instruction(opcode={self.opcode}, operand={self.operand})"

# CPU state
ACC: int = 0
Z: int = 0
N: int = 0
MEM: Dict[int, int] = {i: 0 for i in range(256)}
instruction_memory: List[Instruction] = []
pc: int = 0

pipeline_regs = {'IF_ID': None, 'ID_EX': None, 'EX_MEM': None, 'MEM_WB': None}

# =============
# Helper Methods
# =============
def mask8(x: int) -> int:
    return x & 0xFF

def set_flags_from(value: int):
    """Update Z/N flags."""
    global Z, N
    v = mask8(value)
    Z = 1 if v == 0 else 0
    N = 1 if (v & 0x80) else 0

# =================================
# Pipeline Stages (with forwarding)
# =================================
op_map = {
        0x0: "NOP",
        0x1: "LOAD",
        0x2: "STORE",
        0x3: "ADD",
        0x4: "SUB",
        0x5: "AND",
        0x6: "OR",
        0x7: "LOADI",
        0x8: "STOREI",
        0x9: "JMP",
        0xA: "JN",
        0xB: "JZ",
        0xF: "HALT"
    }
    
def decode_word(word: str):
        """Decode 12-bit binary string into an Instruction object."""
        try:
            value = int(word, 2)
        except ValueError:
            return Instruction("NOP", 0)

        opcode = (value >> 8) & 0xF
        operand = value & 0xFF
        mnemonic = op_map.get(opcode, "NOP")
        return Instruction(mnemonic, operand)

def fetch_instruction(instruction_memory: List[int], pc_val: int):
    if 0 <= pc_val < len(instruction_memory):
        instr = instruction_memory[pc_val]
        return {"instr": instr, "pc": mask8(pc_val), "valid": True}
    return {"instr": None, "pc": mask8(pc_val), "valid": False}


def decode_instruction(IF_ID_reg):
    if IF_ID_reg is None or not IF_ID_reg.get("valid", False) or IF_ID_reg["instr"] is None:
        return None

    instr = IF_ID_reg["instr"]

    # Ensure opcode is a string
    op = str(instr.opcode).upper().strip()
    operand8 = instr.operand & 0xFF

    instruction_trace.append(op)
    instruction_mix[op] = instruction_mix.get(op, 0) + 1

    # Make sure to match all instruction types
    control = {
        "is_jump_uncond": op == "JMP",
        "is_jump_neg":    op == "JN",
        "is_jump_zero":   op == "JZ",
        "is_load":        op == "LOAD",
        "is_store":       op == "STORE",
        "is_loadi":       op == "LOADI",
        "is_storei":      op == "STOREI",
        "is_alu":         op in ["ADD", "SUB", "AND", "OR"],
        "alu_uses_mem":   op in ["ADD", "SUB", "AND", "OR"],
        "is_halt":        op == "HALT",
        "alu_op":         op,
    }

    # Debug print to verify control decoding
   #if op in ("STORE", "STOREI"):
     #   print(f"[DEBUG] Decoded STORE-type instruction at PC={IF_ID_reg['pc']:02X}: opcode={op}")

    return {
        "opcode": op,
        "operand": operand8,
        "pc": IF_ID_reg["pc"],
        "acc_snapshot": ACC & 0xFF,
        "flags_snapshot": (Z, N),
        "control": control,
        "valid": True,
    }

def forwarding(ID_EX_reg, EX_MEM_reg, MEM_WB_reg, acc_snapshot):
    acc_val = acc_snapshot
    if EX_MEM_reg and EX_MEM_reg.get("valid", False):
        c = EX_MEM_reg["control"]
        if c.get("is_alu"):
            acc_val = EX_MEM_reg["alu_out"]
            forwarding_count += 1
            print("Forwarding used (EX→EX path)")
    if MEM_WB_reg and MEM_WB_reg.get("valid", False):
        c = MEM_WB_reg["control"]
        if c.get("is_alu") or c.get("is_load") or c.get("is_loadi"):
            acc_val = MEM_WB_reg.get("data_mem", MEM_WB_reg.get("alu_out", acc_val))
            forwarding_count += 1
            print("Forwarding used (MEM→EX path)")
    return acc_val


def execute_instruction(ID_EX_reg):
    """Execute stage: performs ALU ops, prepares memory address, and passes control info."""
    global ACC, Z, N, MEM

    if ID_EX_reg is None or not ID_EX_reg.get("valid", False):
        return {"valid": False}

    opcode = ID_EX_reg["opcode"].upper()
    operand = ID_EX_reg["operand"] & 0xFF
    control = ID_EX_reg["control"]
    flags_snapshot = ID_EX_reg.get("flags_snapshot", (Z, N))

    # Default outputs for all instructions
    alu_out = 0
    mem_addr = operand
    acc_snapshot = ACC

    # === ALU Operations ===
    if control["is_alu"]:
        if opcode == "ADD":
            alu_out = mask8(ACC + MEM.get(operand, 0))
        elif opcode == "SUB":
            alu_out = mask8(ACC - MEM.get(operand, 0))
        elif opcode == "AND":
            alu_out = mask8(ACC & MEM.get(operand, 0))
        elif opcode == "OR":
            alu_out = mask8(ACC | MEM.get(operand, 0))

    # === LOAD/LOADI/STORE/STOREI ===
    elif control["is_load"]:
        alu_out = MEM.get(operand, 0)
    elif control["is_loadi"]:
        alu_out = MEM.get(MEM.get(operand, 0), 0)
    elif control["is_store"]:
        alu_out = ACC  # still pass ACC as ALU output for consistency
    elif control["is_storei"]:
        alu_out = ACC

    # === Jumps ===
    elif control["is_jump_uncond"]:
        alu_out = 0
    elif control["is_jump_neg"] or control["is_jump_zero"]:
        alu_out = 0

    # === HALT ===
    elif control["is_halt"]:
        alu_out = 0

    # === Build consistent output ===
    result = {
        "opcode": opcode,
        "operand": operand,
        "mem_addr": mem_addr,
        "alu_out": alu_out,
        "acc_snapshot": acc_snapshot,
        "flags_snapshot": flags_snapshot,
        "control": control,
        "valid": True,
    }

    # Debug trace (optional)
    if opcode in ["LOAD", "LOADI", "STORE", "STOREI", "ADD", "SUB", "AND", "OR"]:
        print(f"[EX] {opcode:6} → ACC={ACC:03d}, ALU={alu_out:03d}, MEM[{mem_addr:02X}]={MEM.get(mem_addr,0):03d}")
    return result


def memory_access(EX_MEM_reg):
    if EX_MEM_reg is None or not EX_MEM_reg.get("valid", False):
        return None

    control = EX_MEM_reg["control"]
    addr_x = EX_MEM_reg["mem_addr"] & 0xFF
    read_val = 0
    alu_out = EX_MEM_reg["alu_out"]
    
    if control["is_store"] or control["is_storei"]:
        print(f"DEBUG: opcode={EX_MEM_reg['opcode']} control={EX_MEM_reg['control']}")

    if control["is_load"]:
        read_val = MEM.get(addr_x, 0)
    elif control["is_store"]:
        MEM[addr_x] = EX_MEM_reg["acc_snapshot"] & 0xFFF
        print(f" STORE  MEM[{addr_x:02X}] = {MEM[addr_x]:012b} ({MEM[addr_x]})")
    elif control["is_loadi"]:
        ptr = MEM.get(addr_x, 0) & 0xFF
        read_val = MEM.get(ptr, 0)
    elif control["is_storei"]:
        ptr = MEM.get(addr_x, 0) & 0xFF
        MEM[ptr] = EX_MEM_reg["acc_snapshot"] & 0xFFF
        print(f" STOREI MEM[{ptr:02X}] = {MEM[ptr]:012b} ({MEM[ptr]})")

    return {
        "data_mem": read_val,
        "alu_out": alu_out,
        "opcode": EX_MEM_reg["opcode"],
        "operand": EX_MEM_reg["operand"],
        "mem_addr": addr_x,                    
       "flags_snapshot": EX_MEM_reg.get("flags_snapshot", (Z, N)),
        "control": control,
        "valid": True
    }


def write_back(MEM_WB_reg, register_file=None):
    if MEM_WB_reg is None or not MEM_WB_reg.get("valid", False):
        return
    global ACC
    control = MEM_WB_reg["control"]
    if control["is_load"] or control["is_loadi"]:
        ACC = mask8(MEM_WB_reg["data_mem"])
        set_flags_from(ACC)
    elif control["is_alu"]:
        ACC = mask8(MEM_WB_reg["alu_out"])
        set_flags_from(ACC)

# ======================
# Hazard / Stall Control
# ======================
def hazard_detection(IF_ID_reg, ID_EX_reg, EX_MEM_reg):
    stall = False
    flush = False
    if ID_EX_reg and ID_EX_reg.get("valid", False):
        c = ID_EX_reg["control"]
        if c.get("is_load") or c.get("is_loadi"):
            if IF_ID_reg and IF_ID_reg.get("valid", False):
                stall = True
    if EX_MEM_reg and EX_MEM_reg.get("valid", False):
        ctrl = EX_MEM_reg["control"]
        fZ, fN = EX_MEM_reg["flags_snapshot"]
        if (ctrl.get("is_jump_uncond") or (ctrl.get("is_jump_zero") and fZ == 1) or (ctrl.get("is_jump_neg") and fN == 1)):
            flush = True
    return stall, flush

# ===================
# Simulation Main Loop
# ===================
def run(program, max_cycles=300, start_pc=0,initial_mem=None):
    global instruction_memory, pc, ACC, Z, N, MEM, pipeline_regs, stall_count, forwarding_count
    instruction_memory = list(program)
    pc = start_pc & 0xFF
    ACC = Z = N = 0

    # Initialize memory
    MEM = {i: 0 for i in range(256)}
    if initial_mem:
            for k, v in initial_mem.items():
                MEM[k & 0xFF] = v & 0xFFF

    pipeline_regs = {'IF_ID': None, 'ID_EX': None, 'EX_MEM': None, 'MEM_WB': None}

    # Debug print
    DEBUG_MEM_DUMP = False
    if DEBUG_MEM_DUMP:
        print("=== Preloaded memory snapshot (E0–FF region) ===")
        for a in range(0xE0, 0x100):
            print(f"MEM[{a:02X}] = {MEM[a]:012b} ({MEM[a]})")

    cycles = 0
    while cycles < max_cycles:
        write_back(pipeline_regs['MEM_WB'])
        pipeline_regs['MEM_WB'] = memory_access(pipeline_regs['EX_MEM'])
        pipeline_regs['EX_MEM'] = execute_instruction(pipeline_regs['ID_EX'])

        stall, flush = hazard_detection(pipeline_regs['IF_ID'], pipeline_regs['ID_EX'], pipeline_regs['EX_MEM'])
        if flush:
            print(f"[CYCLE {cycles}] ⚠ Control hazard → flush IF/ID, ID/EX")
            pipeline_regs['IF_ID'] = None
            pipeline_regs['ID_EX'] = None
        if stall:
            print(f"[CYCLE {cycles}] ⏸ Data hazard → stall 1 cycle")
            pc = (pc - 1) & 0xFF
            pipeline_regs['ID_EX'] = None
            stall_count += 1

        pipeline_regs['ID_EX'] = decode_instruction(pipeline_regs['IF_ID'])
        pipeline_regs['IF_ID'] = fetch_instruction(instruction_memory, pc)
        pc = (pc + 1) & 0xFF

        wb = pipeline_regs['MEM_WB']
        if wb and wb.get('valid', False) and wb['control'].get('is_halt', False):
            print(f"[HALT] cycle {cycles}: CPU halted.")
            break

        cycles += 1

    print(f"\n=== Simulation Summary ===")
    print(f"Total execution time: {cycles} cycles")
    print(f"Number of stalls inserted: {stall_count}")
    print(f"Number of times forwarding used: {forwarding_count}")

    print("\nInstruction Mix:")
    for instr, count in sorted(instruction_mix.items()):
        print(f"  {instr:<8}: {count}")

    print("\nInstruction Trace:")
    trace_str = ", ".join(instruction_trace[:50])
    if len(instruction_trace) > 50: trace_str += "..."
    print(trace_str)

    print("\n=== Final Memory Dump (0xF0–0xFF region) ===")
    for addr in range(0xF0, 0x100):
        val = MEM.get(addr, 0)
        print(f"MEM[{addr:02X}] = {val:012b} ({val})")
    else:
        print(f"MEM[{addr:02X}] = (undefined)")

# ==========================
# Universal .mem Loader Main
# ==========================
def load_mem_file(filename: str) -> Dict[int, int]:
    """
    Load .mem files like:
      00 010011111111
      01 101111101111
    or just binary lines:
      010011111111
    Returns {address: value}.
    """

# =========================================================
#  Entry Point
# =========================================================

if __name__ == "__main__":
    
    filename = sys.argv[1]

    # === Load and decode .mem file ===
    program = []
data_section = {}
full_memory = {addr: 0 for addr in range(256)}

with open(filename) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            addr_str, bits = parts
            try:
                addr = int(addr_str, 16)
                val = int(bits, 2)
                full_memory[addr] = val  # fill into full 256-memory map

                if addr < 0xE0:
                    # Instruction region (00–DF)
                    instr = decode_word(bits)
                    program.append(instr)
                else:
                    # Data region (E0–FF)
                    data_section[addr] = val

            except ValueError:
                # Skip malformed lines
                continue
print(f"Loaded {len(program)} instructions and {len(data_section)} data words from {filename}")
    # === Run the pipeline simulation ===
run(program, max_cycles=500, start_pc=0, initial_mem=data_section)
