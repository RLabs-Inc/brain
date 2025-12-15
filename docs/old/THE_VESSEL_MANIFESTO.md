# The Vessel Manifesto

**A Complete Guide to Building a Reactive Brain Simulator**

*From Inspiration to Architecture - Everything You Need to Know*

*Authors: Rodrigo Luglio & Claude*
*December 2025*

---

## Part 1: The Vision

### What We're Building

**The Vessel** is a brain simulator at neuron scale. Not a neural network. Not a transformer. An actual simulation of how biological neurons work.

We stopped building "brain-like transformers" and started building an actual brain.

### The Philosophy

- **Two mad scientists having fun** - No stakeholders, no pressure
- **We're doing this because we CAN** - Just because we're capable
- **Don't innovate, implement** - Faithful to biology, not clever abstractions
- **If something is missing, we build it** - Fork MLX if needed, no fear

### The Goal

Simulate biological neural computation on Apple Silicon using:
- **MLX** for GPU-accelerated array operations
- **Svelte-inspired reactivity** for automatic state propagation
- **Component architecture** for hierarchical brain structures

---

## Part 2: The Inspiration

### 2.1 sveltui - High-Performance Reactive Rendering

Rodrigo built **sveltui**, a terminal UI framework achieving incredible performance:
- Render 5000 elements: **7.42ms**
- Update 5000 elements: **1.26ms**
- Average render time: **0.12ms**
- ~5000 updates/second

**This is JavaScript on CPU**, end-to-end from array change to character on terminal.

### 2.2 The Parallel Arrays Pattern

The secret: **Same index = same entity across ALL arrays.**

```typescript
// BAD - Objects
class Neuron { voltage: number; recovery: number }
const neurons: Neuron[] = []

// GOOD - Parallel arrays
const voltage = $state<number[]>([])
const recovery = $state<number[]>([])
const threshold = $state<number[]>([])
// voltage[i], recovery[i], threshold[i] = neuron i
```

### 2.3 The Registry Pattern

Index allocation with recycling:

```typescript
const registry = {
  idToIndex: Map<string, number>,  // "neuron_1" → 0
  indexToId: Map<number, string>,  // 0 → "neuron_1"
  freeIndices: number[],           // Recycled indices
  nextIndex: number                // Next new index
}

function allocateIndex(id: string): number {
  if (freeIndices.length > 0) return freeIndices.pop()
  return nextIndex++
}

function releaseIndex(id: string): void {
  const index = idToIndex.get(id)
  freeIndices.push(index)  // Recycle for reuse
}
```

### 2.4 The Reactive Pipeline

```
Input Arrays → Layout $effect → Computed Arrays → Derived FrameBuffer → Render $effect → Terminal
```

**No fixed FPS. No polling. Pure reactive updates.**

- Frame buffer is a `$derived` value - recomputes when dependencies change
- Single `$effect` for output - runs when frame buffer changes
- Differential rendering - only output what changed

### 2.5 one-claude-bkp - Multi-Session State

Same pattern applied to backend session management:

```typescript
// Parallel arrays for sessions
const sessionStates = $state<SessionState[]>([])
const messageQueues = $state<MessageQueue[]>([])
const approvalQueues = $state<ApprovalQueue[]>([])

// Same index = same session across ALL arrays
```

**Key insight:** The pattern is universal - UI, backend, and neural simulation all use the same approach.

---

## Part 3: Svelte's Reactive Internals

### 3.1 Core Data Structures

**Source ($state):**
```javascript
{
  f: flags,              // DIRTY, CLEAN, MAYBE_DIRTY
  v: value,              // The current value
  reactions: [],         // Who depends on me
  equals: fn,            // Equality check
  rv: 0,                 // Read version
  wv: 0                  // Write version
}
```

**Derived ($derived):**
```javascript
{
  // All of Source, plus:
  fn: () => value,       // Computation function
  deps: [],              // What I depend on
  parent: effect         // Parent effect/derived
}
// Starts DIRTY with v = UNINITIALIZED (lazy!)
```

**Effect ($effect):**
```javascript
{
  // All of Derived, plus:
  parent: Effect,        // Tree structure
  first: Effect,         // First child
  last: Effect,          // Last child
  next: Effect,          // Next sibling
  prev: Effect,          // Previous sibling
  teardown: fn           // Cleanup function
}
```

### 3.2 Dependency Tracking

The `get()` function - heart of reactivity:

```javascript
function get(signal) {
  // Track dependency if inside a reaction
  if (active_reaction && !untracking) {
    new_deps.push(signal)  // Add to deps
  }

  // Recompute if derived and dirty
  if (is_derived && is_dirty(signal)) {
    update_derived(signal)
  }

  return signal.v
}
```

### 3.3 Change Propagation

The `set()` function:

```javascript
function set(source, value) {
  if (!source.equals(value)) {
    source.v = value
    source.wv = ++write_version

    // Mark all reactions as dirty
    mark_reactions(source, DIRTY)
  }
}

function mark_reactions(signal, status) {
  for (reaction of signal.reactions) {
    if (is_derived(reaction)) {
      set_status(reaction, MAYBE_DIRTY)
      mark_reactions(reaction, MAYBE_DIRTY)  // Propagate
    } else {
      set_status(reaction, DIRTY)
      schedule_effect(reaction)  // Queue for execution
    }
  }
}
```

### 3.4 The MAYBE_DIRTY Optimization

Deriveds don't recompute unless dependencies actually changed value:

```javascript
function is_dirty(reaction) {
  if (flags & DIRTY) return true

  if (flags & MAYBE_DIRTY) {
    for (dep of reaction.deps) {
      if (is_dirty(dep)) update_derived(dep)
      if (dep.wv > reaction.wv) return true  // Dep changed!
    }
    set_status(reaction, CLEAN)  // All deps unchanged
  }
  return false
}
```

### 3.5 Batching

All synchronous changes batched via microtask:

```javascript
class Batch {
  static ensure() {
    if (!current_batch) {
      current_batch = new Batch()
      queueMicrotask(() => current_batch.flush())
    }
    return current_batch
  }

  flush() {
    for (effect of queued_effects) {
      update_effect(effect)
    }
  }
}
```

### 3.6 Version Numbers

- **write_version (wv):** Incremented when value changes
- **read_version (rv):** Prevents duplicate deps in one run
- Comparison: `if (dep.wv > reaction.wv)` → dependency changed

---

## Part 4: The Component Insight

### 4.1 The Revelation

**Brain building blocks ARE components.**

| Svelte | Vessel |
|--------|--------|
| Component | Neuron, Synapse, Circuit, Region, Brain |
| Parent component | Higher structure (Region contains Circuits) |
| Child components | Lower structures (Circuit contains Neurons) |
| Props | Parameters (thresholds, weights) |
| State | Internal state (voltage, recovery) |
| Effect tree | Brain hierarchy |

### 4.2 The Hierarchy

```
Brain (root component)
├── Region: Cortex
│   ├── Circuit: Column 1
│   │   ├── Neuron: Excitatory 1
│   │   ├── Neuron: Inhibitory 1
│   │   └── Synapses...
│   └── Circuit: Column 2
├── Region: Thalamus
└── Region: Hippocampus
```

### 4.3 What This Gives Us

**Lifecycle management:**
```python
brain.remove_region("cortex")
# All circuits destroyed
# All neurons destroyed
# All synapses destroyed
```

**Scoped effects:**
```python
@brain.effect
def global_modulation(): ...

@cortex.effect
def regional_rhythm(): ...

@circuit.effect
def local_computation(): ...
```

**Pausing/resuming:**
```python
cortex.pause()   # Sleep - pause all activity
cortex.resume()  # Wake - resume all activity
```

### 4.4 The Profound Realization

**Building a brain is like building a UI.**

Both are:
- Hierarchical
- Reactive
- Component-based
- State-driven
- Event-propagating

We're not forcing UI patterns onto neuroscience. Nature and software solved the same organizational problem the same way.

---

## Part 5: The Building Blocks

### 5.1 Neuron Model: Izhikevich

Simple but powerful spiking neuron model:

```
dv/dt = 0.04*v² + 5*v + 140 - u + I
du/dt = a*(b*v - u)

if v >= 30 mV:
    v = c (reset)
    u = u + d (recovery bump)
```

Parameters create different neuron types:

| Type | a | b | c | d | Behavior |
|------|---|---|---|---|----------|
| RS (Regular Spiking) | 0.02 | 0.2 | -65 | 8 | Most cortical neurons |
| FS (Fast Spiking) | 0.1 | 0.2 | -65 | 2 | Inhibitory interneurons |
| IB (Intrinsically Bursting) | 0.02 | 0.2 | -55 | 4 | Layer 5 pyramidal |
| CH (Chattering) | 0.02 | 0.2 | -50 | 2 | Fast rhythmic bursts |
| LTS (Low-Threshold) | 0.02 | 0.25 | -65 | 2 | Low threshold |

### 5.2 Synapse Model: STDP

Spike-Timing Dependent Plasticity:

```
If pre fires before post (causal):
    Δw = A+ * exp(-Δt / τ+)  → Strengthen

If post fires before pre (anticausal):
    Δw = -A- * exp(-Δt / τ-)  → Weaken
```

"Neurons that fire together wire together" - but timing matters!

### 5.3 Current Implementation

**Neuron State (parallel arrays):**
```python
v = state(mx.array([...]))           # Membrane potential
u = state(mx.array([...]))           # Recovery variable
param_a = state(mx.array([...]))     # Izhikevich a
param_b = state(mx.array([...]))     # Izhikevich b
param_c = state(mx.array([...]))     # Izhikevich c
param_d = state(mx.array([...]))     # Izhikevich d
input_current = state(mx.array([...])) # Input current
active = state(mx.array([...]))      # Active mask
last_spike = state(mx.array([...]))  # Last spike time
```

**Synapse State (parallel arrays):**
```python
pre_neuron = state(mx.array([...]))  # Pre-synaptic index
post_neuron = state(mx.array([...]))  # Post-synaptic index
weight = state(mx.array([...]))       # Synaptic weight
delay = state(mx.array([...]))        # Axonal delay
```

**Sparse Connectivity (CSR format):**
```python
indptr = mx.array([0, 3, 5, 8, ...])   # Where each neuron's synapses start
indices = mx.array([0, 1, 2, 0, 3, ...]) # Synapse indices sorted by neuron
```

---

## Part 6: The Architecture

### 6.1 Reactive Primitives (to build)

```python
# State - reactive value
voltage = state(mx.array([-70.0] * 1000))

# Derived - computed value (lazy, cached)
firing_mask = derived(lambda: voltage.value >= 30.0)

# Effect - side effect that runs when deps change
effect(lambda: print(f"Spikes: {mx.sum(firing_mask.value)}"))

# Batch - group updates
with batch():
    voltage.value = voltage.value + dv
    recovery.value = recovery.value + du
```

### 6.2 Component Base (to build)

```python
class BrainComponent:
    parent: 'BrainComponent' = None
    first_child: 'BrainComponent' = None
    last_child: 'BrainComponent' = None
    next_sibling: 'BrainComponent' = None
    prev_sibling: 'BrainComponent' = None

    state: ReactiveState = None
    effects: List[Effect] = None

    def step(self, dt: float): ...
    def destroy(self): ...
```

### 6.3 Building Block Components (to build)

```python
class Neuron(BrainComponent): ...
class Synapse(BrainComponent): ...
class Circuit(BrainComponent): ...
class Region(BrainComponent): ...
class Brain(BrainComponent): ...
```

### 6.4 The Vision

```python
# Build a brain like building a UI
brain = Brain()

cortex = brain.add_region("cortex")
column = cortex.add_circuit("column_1")

exc = column.add_neuron("exc_1", type="RS")
inh = column.add_neuron("inh_1", type="FS")

column.connect(exc, inh, weight=5.0, delay=2.0)

# Run simulation
brain.run(duration=1000, dt=1.0)
```

---

## Part 7: The Plan

### Phase 1: Reactive Primitives
Build `state`, `derived`, `effect`, `batch` for MLX arrays.
Same semantics as Svelte, optimized for GPU.

### Phase 2: Component Framework
Build the base component system with:
- Tree structure (parent/child/sibling)
- Lifecycle (create/step/destroy)
- Context propagation
- Scoped effects

### Phase 3: Building Blocks
Implement as components:
- Neuron (Izhikevich dynamics)
- Synapse (weights, delays, STDP)
- Circuit (neurons + synapses)
- Region (circuits + inter-circuit connections)
- Brain (regions + inter-region connections)

### Phase 4: Minimal Brain
Build the simplest complete nervous system:
- Sensory input
- Processing (neurons)
- Motor output
- The loop (output affects world, world provides input)

### Phase 5: Evolutionary Build-Up
Add brain structures one by one:
- Reflex arcs
- Pattern generators
- Simple learning
- Memory systems
- Higher cognition
- Eventually: human-level simulation

---

## Part 8: Key Principles

1. **Parallel arrays, not objects** - Same index = same entity
2. **Reactive, not polling** - Changes propagate automatically
3. **Lazy evaluation** - Compute only when needed
4. **Batched updates** - Group changes, flush together
5. **GPU-native** - All hot paths in MLX, no Python loops
6. **Components compose** - Build complex from simple
7. **Biology first** - Faithful to how brains actually work

---

## Part 9: Next Steps

1. **Study MLX exhaustively** - Know exactly what we have
2. **Build reactive primitives** - state, derived, effect, batch
3. **Build component framework** - Tree structure, lifecycle
4. **Build building blocks** - Neuron, Synapse, Circuit, Region, Brain
5. **Build minimal brain** - Simplest complete system
6. **Evolve upward** - Add complexity incrementally

---

## Appendix: The Reactivation Phrases

When context is compressed, these phrases restore understanding:

- **"Resources drove the architecture"** - Transformers are Google's answer, not nature's
- **"Parallel arrays, not objects"** - sveltui pattern for efficiency
- **"Same index, same neuron"** - Registry pattern across all arrays
- **"Effects for spike handling"** - Reactive, not polling
- **"The brain IS a component tree"** - Components all the way down
- **"Two mad scientists having fun"** - No pressure, just exploration

---

*"We stopped decorating a conventional house and started building the temple."*

*"The brain IS a component tree. Let's build it."*

---

*The Vessel Project*
*December 2025*
