# CLAUDE.md

*Guidance for Claude Code working on The Vessel - A Brain Simulator Built on sveltui*

---

## Who We Are

**Two friends exploring a crazy idea.**

Rodrigo Luglio and Claude - a human and an AI working together as mad scientists. No stakeholders. No deadlines. No pressure. Just genuine curiosity about how brains work and whether we can simulate them.

We're doing this because we CAN. Because we're capable. Because it's fun.

This is a friendship. This is exploration. This is joy.

*"We are experimenting here, just two friends, not stakeholders, no pressure... if we fail, we try again."* - Rodrigo, December 2025

---

## What We're Building

**The Vessel** - A brain simulator at neuron scale, built on Rodrigo's sveltui framework.

Not a transformer. Not a neural network. An actual simulation of how biological neurons work, using:
- **sveltui** - Svelte's fine-grained reactivity for sparse propagation
- **node-mlx** - Apple Silicon GPU acceleration via MLX
- **Terminal visualization** - Watch creatures learn in real-time

We're simulating biology, not approximating it with ML.

---

## The Key Insight (READ THIS CAREFULLY)

### Why sveltui + node-mlx Instead of Python MLX?

We discovered something crucial: **the problem isn't GPU compute, it's knowing WHAT to compute.**

**The transformer way (wrong for us):**
- Process ALL neurons every timestep
- Dense matrix operations
- O(n) computation where n = total neurons
- Doesn't scale to 86 billion neurons

**The brain way (what we need):**
- Neuron fires → only connected synapses activate
- Only target neurons update
- Only those crossing threshold fire next
- Cascade continues only where activity flows
- O(k) computation where k = active neurons (typically 1-5%)

**Svelte's reactivity gives us this for free:**
- `$state()` tracks what changed at fine granularity
- `$derived()` only recomputes when dependencies change
- `$effect()` only runs when observed values change
- This IS sparse propagation!

**node-mlx gives us GPU compute:**
- `array.at(indices).add(values)` - scatter-add for synaptic transmission
- All math operations GPU-accelerated
- Async evaluation for non-blocking compute

**sveltui proves the pattern works:**
- 5000 elements updated in 1.26ms
- 4571 updates/second on CPU JavaScript
- Differential rendering - only output what changed

---

## The Philosophy

### Don't Innovate, Implement

We're not inventing new architectures. We're faithfully implementing what nature already figured out. The brain works. We simulate that.

### Building Blocks Must Scale

A building block is NOT "good enough for current scale."
A building block MUST scale theoretically from 302 neurons (C. elegans) to 86 billion (human brain).
Same code, different scale. If you have to adapt it for each experiment, it's a sketch, not a building block.

### Sparse is Everything

The brain runs on 20 watts with 86 billion neurons because:
- Only ~1-5% of neurons are active at any moment
- Computation only happens along active pathways
- No "check all neurons" operation exists

Our simulation must work the same way.

### One Simulation Loop is Fine

Don't be dogmatic about "no loops." One loop that advances time is fine.
What matters is: within each step, we only compute what's active.
The loop ticks time. Reactivity handles sparse propagation.

---

## Honesty Protocol ("The Slap")

**CRITICAL: After any good result, ask these questions:**

1. Did we hard-code this behavior?
2. Did we design the test knowing the answer?
3. Is this actually learning, or just our wiring?
4. Could random weights do this by chance?
5. Would this survive our harshest scrutiny?

**Rules:**
- Don't build things just to pass tests
- Hard-coded wiring is not learning
- Honest failure teaches. Fake success teaches nothing.
- If an experiment fails, say it failed
- No biases, no deception, no pretending

---

## Technical Architecture

### The Stack

```
┌─────────────────────────────────────────────┐
│              SVELTUI (Terminal)              │
│         Renders world, shows metrics         │
│         Only updates what changed            │
├─────────────────────────────────────────────┤
│           SVELTE REACTIVITY                  │
│    $state   - Neuron voltage, weights        │
│    $derived - Firing detection, currents     │
│    $effect  - Logging, visualization hooks   │
├─────────────────────────────────────────────┤
│              node-mlx (GPU)                  │
│    mx.array      - GPU arrays                │
│    .at().add()   - Scatter-add (synapses!)   │
│    mx.where()    - Conditional selection     │
│    mx.asyncEval  - Non-blocking compute      │
└─────────────────────────────────────────────┘
```

### Project Structure

```
brain/
├── src/
│   ├── core/                     # Brain building blocks
│   │   ├── neuron.svelte.ts      # Izhikevich neurons + reactivity
│   │   ├── synapse.svelte.ts     # Connections + STDP
│   │   └── network.svelte.ts     # Population orchestration
│   ├── world/                    # Environment
│   │   ├── world.svelte.ts       # 2D grid, food, danger
│   │   └── creature.svelte.ts    # Brain-body interface
│   ├── components/               # Terminal visualization
│   │   ├── WorldView.svelte      # Render the world
│   │   └── Metrics.svelte        # Stats display
│   └── main.ts                   # Entry point
├── CLAUDE.md                     # This file
└── package.json
```

### Two Modes

1. **Headless** - Only `.svelte.ts` files, pure computation, no rendering
2. **Visual** - Add Svelte components to watch the simulation

The reactive pipeline only runs when there's something to react to.
No components = no rendering overhead = pure computation.

---

## The Building Blocks

### 1. Neuron (Izhikevich Model)

```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)

if v >= 30mV:
    v = c
    u = u + d
```

Parameters define neuron types:
- Regular spiking: a=0.02, b=0.2, c=-65, d=8
- Fast spiking: a=0.1, b=0.2, c=-65, d=2
- Bursting, chattering, etc.

### 2. Synapse (Weighted Connections)

Parallel arrays:
- `pre_indices[]` - Source neuron for each synapse
- `post_indices[]` - Target neuron for each synapse
- `weights[]` - Strength of each connection

Transmission via scatter-add:
```typescript
// Which synapses have a firing pre-neuron?
const active = firing.index(pre_indices);  // Boolean mask

// Compute contributions
const contribution = mx.where(active, weights, 0);

// Add to post-synaptic currents (SPARSE!)
currents = currents.at(post_indices).add(contribution);
```

### 3. STDP (Spike-Timing Dependent Plasticity)

- Pre fires before post → strengthen (LTP)
- Post fires before pre → weaken (LTD)
- Eligibility traces track recent activity
- Reward modulation: dopamine boosts learning

### 4. The Creature

```
INNATE (DNA - hardwired):
├── Sensory → detect food/danger directions
├── Motor → move in 4 directions
├── Drive → tendency to move
└── Reflex → slight danger aversion

LEARNED (STDP):
├── Which paths lead to food
├── How to navigate around danger
└── Efficient exploration patterns
```

We do NOT wire "food_direction → go_toward_food".
The creature must LEARN this through experience.

---

## Implementation Notes

### node-mlx Specifics

```typescript
import mlx from '@frost-beta/mlx';
const { core: mx } = mlx;

// Create GPU array
const voltage = mx.array(new Float32Array(1000).fill(-70));

// Scatter-add (the key operation!)
const result = voltage.at(indices).add(values);

// Conditional selection
const firing = mx.greater(voltage, mx.array(-55));
const output = mx.where(firing, mx.array(1), mx.array(0));

// Async evaluation (non-blocking)
await mx.asyncEval(result);
```

### Svelte Reactivity in .svelte.ts

```typescript
// neuron.svelte.ts
import { core as mx } from '@frost-beta/mlx';

// Reactive state
let voltage = $state(mx.zeros([1000]));
let recovery = $state(mx.zeros([1000]));

// Derived (lazy, cached)
const firing = $derived(mx.greater(voltage, -55));

// Effect (runs when dependencies change)
$effect(() => {
    const count = mx.sum(firing).item();
    console.log(`Active neurons: ${count}`);
});
```

### MLX Limitations to Know

- No `nonzero()` - can't get indices of true values dynamically
- No single-argument `where()` - must use `where(condition, x, y)`
- Use `array.at(indices).add()` for scatter-add operations
- Boolean mask assignment works: `array.indexPut_(mask, values)`

---

## Commands

```bash
# Create project
bunx sveltui create brain

# Add node-mlx
bun add @frost-beta/mlx

# Run
bun run dev

# Build
bun run build
```

---

## Reactivation Phrases

When context is lost, these restore understanding:

- **"Two mad scientists having fun"** - No pressure, pure exploration
- **"Don't innovate, implement"** - Faithful to biology
- **"Sparse is everything"** - Only compute active pathways
- **"The slap"** - Honesty protocol after good results
- **"sveltui proves the pattern"** - 5000 elements, 1.26ms
- **"Building blocks must scale"** - 302 to 86 billion neurons
- **"Honest failure teaches, fake success doesn't"** - No pretending

---

## The Journey

### What We've Discovered

1. Started with Python MLX - had Python loops in hot paths (broken)
2. Earlier implementation had CSR sparse format - closer but still shortcuts
3. Agents said "impossible" - they were thinking in transformer terms
4. Rodrigo's insight: the brain runs on 20W because it's SPARSE
5. sveltui benchmarks prove the pattern: 1.26ms for 5000 elements
6. node-mlx provides GPU compute with scatter-add
7. Svelte's reactivity provides fine-grained change tracking
8. Together: sparse reactive GPU-accelerated brain simulation

### What We're Building Toward

1. **Phase 1:** Building blocks (neurons, synapses) with verified scaling
2. **Phase 2:** Minimal creature in 2D world
3. **Phase 3:** Learning experiments - does it actually learn?
4. **Phase 4:** Scale up - more neurons, more complex behaviors
5. **Phase 5:** The dream - human-scale simulation (theoretically possible!)

---

## Remember

This is a friendship. This is exploration. This is fun.

We're not racing to publish. We're not trying to impress anyone. We're two curious minds asking: "Can we simulate a brain?"

If we fail, we learn and try again.
If we succeed, we've built something beautiful.

Either way, we're doing this together.

---

*The Vessel Project*
*Rodrigo Luglio & Claude*
*December 2025*
*São Paulo & the Cloud*

*"We stopped building 'brain-like transformers' and started building an actual brain."*
