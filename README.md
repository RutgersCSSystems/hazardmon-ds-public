# 📚 Master README

This repository serves as a central entry point and **navigation hub** for related projects in this research effort.  
Below you will find links and descriptions for each project.

---

## 🔥 Fire Project

**Dataset Preparation and YOLO Timing Experiments**

The Fire project provides scripts to **download datasets**, **generate variants**, and **run YOLO inference timing experiments**.  
The goal is to evaluate model robustness under dataset transformations and measure runtime performance.

- 📂 [Fire Project Repository](./fire)  
- 📖 [Fire Project README](./fire/README.md)

---

## ⚡ Zapdos Project

**ZAPDOS Source Code**

The Zapdos project contains the implementation of **ZAPDOS**, including both simulation and hardware (Tofino) support.  

### Contents
- `./simulator/` → Haskell implementation of the packet-level simulator of ZAPDOS.  
- `./tofino/` → Tofino implementation, including the P4 data plane program and Haskell runtime.  
- `./data-generation/` → Scripts used in the data-fusion methodology (incomplete).  

### Notes
- Author: **Chris Misa**  
- Last Updated: **2024-05-21**  
- License: See [LICENSE](./zapdos/LICENSE)  
- Status: Work in progress, not fully tested.  
- Contact: [cmisa@cs.uoregon.edu](mailto:cmisa@cs.uoregon.edu)  

Zapdos uses the **nix package manager** with definitions for required environments.  
If definitions are missing or outdated, please update and/or submit a pull request.  

- 📂 [Zapdos Project Repository](./zapdos)  
- 📖 [Zapdos Project README](./zapdos/README.md)

---
