# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Vessel is a brain simulator at neuron scale built on **sveltui** (terminal UI framework). It uses Svelte's fine-grained reactivity for sparse propagation and **node-mlx** for Apple Silicon GPU acceleration.

The core insight: the brain runs on 20W with 86B neurons because only ~1-5% are active at any moment. We leverage Svelte's `$state`/`$derived` reactivity to compute only along active pathways, combined with GPU scatter-add operations for synaptic transmission.

## Commands

```bash
bun install           # Install dependencies
bun run dev           # Build and run (equivalent to build + start)
bun run build         # Compile Svelte + sveltui files to dist/
bun run start         # Run the compiled application
```

The build script (`./sveltui-build`) compiles all `.svelte` and `.svelte.ts` files together with the sveltui framework in a single pass for proper reactivity.

## Architecture

### Tech Stack
- **sveltui**: Terminal UI framework using Svelte 5 reactivity + Yoga layout
- **node-mlx** (`@frost-beta/mlx`): MLX bindings for Apple Silicon GPU
- **Svelte 5**: Runes (`$state`, `$derived`, `$effect`) for fine-grained reactivity

### Core Building Blocks (src/core/)

All follow the sveltui pattern with direct `$state` exports and SvelteMap registries:

- **neuron.svelte.ts**: Izhikevich neuron populations
  - Direct exports: `voltage`, `recovery`, `current` (GPU arrays indexed by population)
  - Functions: `allocatePopulation()`, `integrate()`, `injectCurrent()`
  - Double-function pattern for derived: `getNeuronDerived(popIndex)()`

- **synapse.svelte.ts**: Sparse synaptic transmission with STDP
  - Key operation: scatter-add via `current.at(postIndices).add(contribution)`
  - Functions: `transmit()`, `updateTraces()`, `applySTDP()`, `applyReward()`
  - Connectivity helpers: `createAllToAllConnectivity()`, `createRandomConnectivity()`

- **network.svelte.ts**: Orchestration layer
  - Groups populations and synapse groups
  - Global dopamine signal for three-factor learning
  - `step(networkIndex, dt)` - called by external driver, not time-controlling

### Key Patterns

1. **ALL computation on GPU** - No `.item()` or `.tolist()` except at final output
2. **JS loops only for structure** - Iterating over populations/groups (small numbers), not neurons
3. **Double-function for derived** - `getNeuronDerived(index)()` returns reactive GPU values
4. **Scatter-add for sparse ops** - `array.at(indices).add(values)` is the core operation

### MLX Usage

```typescript
import { core as mx } from '@frost-beta/mlx'

// GPU arrays
const v = mx.full([size], -70, mx.float32)

// Scatter-add (THE key operation for sparse transmission)
current = current.at(postIndices).add(contribution)

// Conditional selection
const fired = mx.greaterEqual(v, mx.array(30))
voltage = mx.where(fired, resetValue, voltage)

// Async evaluation
await mx.asyncEval(voltage, recovery)
```

**MLX Limitations:**
- No `nonzero()` - can't get indices of true values dynamically
- No single-argument `where()` - must use `where(condition, x, y)`
- Boolean mask indexing works: `array.index(boolMask)`

### MLX Memory Management (CRITICAL!)

MLX arrays are lazy-evaluated and each `mx.array()` allocates a Metal buffer. Without proper cleanup, long-running simulations exhaust GPU resources (~1.86GB before crash on M1 Max).

**The Golden Rules:**

1. **Cache constants at module level** - Never create the same array repeatedly:
```typescript
// GOOD - created once, reused forever
const CONST_ZERO = mx.array(0, mx.float32)
const CONST_ONE = mx.array(1, mx.float32)
const CONST_EPSILON = mx.array(1e-8, mx.float32)

// BAD - creates new Metal buffer every call!
function compute() {
  const zero = mx.array(0, mx.float32)  // LEAK!
}
```

2. **Cache arrays used in hot loops** - Especially in `think()` or per-timestep functions:
```typescript
// GOOD - cache at creature creation
const cachedDrive = mx.full([size], 12.0, mx.float32)
const cachedIndices = mx.arange(0, size, 1, mx.int32)

function think() {
  injectCurrent(pop, cachedIndices, cachedDrive)  // Reuse!
}

// BAD - creates 4 GPU arrays every timestep!
function think() {
  const drive = mx.full([size], 12.0, mx.float32)  // LEAK!
  injectCurrent(pop, mx.arange(0, size, 1, mx.int32), drive)  // LEAK!
}
```

3. **Wrap computations in mx.tidy()** - Cleans up intermediate tensors:
```typescript
// GOOD - intermediates cleaned up automatically
const result = mx.tidy(() => {
  const a = mx.multiply(x, y)       // intermediate - cleaned
  const b = mx.add(a, z)            // intermediate - cleaned
  const c = mx.divide(b, scale)     // intermediate - cleaned
  mx.eval(c)                        // IMPORTANT: eval INSIDE tidy!
  return c                          // returned - NOT cleaned (we need it)
})

// BAD - all intermediates leak!
const a = mx.multiply(x, y)         // LEAK!
const b = mx.add(a, z)              // LEAK!
const c = mx.divide(b, scale)       // LEAK!
```

4. **mx.eval() MUST be inside tidy** - Or you get "Cannot access stream on invalid event":
```typescript
// GOOD
const result = mx.tidy(() => {
  const computed = mx.add(x, y)
  mx.eval(computed)  // Inside tidy!
  return computed
})

// BAD - will error!
const result = mx.tidy(() => {
  return mx.add(x, y)
})
mx.eval(result)  // Error: invalid event
```

5. **State updates happen OUTSIDE tidy** - Return values, then assign:
```typescript
// GOOD
const result = mx.tidy(() => {
  const newValue = mx.add(state[i], delta)
  mx.eval(newValue)
  return newValue
})
state[i] = result  // Update outside

// BAD - modifying state inside tidy can cause issues
mx.tidy(() => {
  state[i] = mx.add(state[i], delta)  // Risky!
})
```

**Memory Leak Checklist:**
- [ ] All constants cached at module top level?
- [ ] Hot-loop arrays (think, sense, act) cached at creation time?
- [ ] All GPU computation paths wrapped in mx.tidy()?
- [ ] mx.eval() called INSIDE tidy blocks?
- [ ] State assignments happen OUTSIDE tidy?

## Philosophy

- **Don't innovate, implement**: Faithfully simulate biology (Izhikevich model, STDP)
- **Sparse is everything**: Only compute active pathways
- **Building blocks must scale**: Same code from 302 neurons (C. elegans) to 86B (human)
- **One simulation loop is fine**: Loop ticks time, reactivity handles sparse propagation within each step

## Honesty Protocol ("The Slap")

We are doing real science here. After ANY good result, ask these questions:

1. Did we hard-code this behavior?
2. Did we design the test knowing the answer?
3. Is this actually learning, or just our wiring?
4. Could random weights do this by chance?
5. Would this survive our harshest scrutiny?

**Rules:**
- If an experiment fails, say it failed. Honest failure teaches - fake success teaches nothing.
- Never build tests or simulations designed to produce the results we want.
- Never hard-code good results or cherry-pick successful runs.
- All tests run always - no skipping inconvenient ones.
- No biases, no deception, no pretending.

If we fail, we learn and start again. That's how real discovery works.

## Detailed Documentation

See `docs/CLAUDE.md` for the full project manifesto, journey context, and reactivation phrases.
See `docs/ARCHITECTURE.md` for detailed technical architecture.
