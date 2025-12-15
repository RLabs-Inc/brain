# The Vessel Architecture

*Technical deep-dive into sparse reactive neural simulation*

---

## The Core Problem

Traditional neural simulators (NEURON, Brian2, etc.) use a **tick-based approach**:

```
for each timestep:
    for each neuron:
        update voltage
        check if firing
    for each synapse:
        if pre fired: transmit
```

This is O(n) per timestep where n = total neurons. It doesn't scale.

The brain has 86 billion neurons but runs on 20 watts. How?

**Answer: Sparse, event-driven computation.**

Only ~1-5% of neurons are active at any moment. The brain doesn't "check all neurons" - activity propagates only along active pathways.

---

## Our Solution: Reactive Sparse Propagation

### The Insight

Svelte's fine-grained reactivity gives us exactly what we need:

```
State changes → Only dependent computations run → Only affected outputs update
```

This IS sparse propagation! The same pattern that makes sveltui fast (5000 elements, 1.26ms) can make neural simulation efficient.

### The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     OBSERVATION LAYER                        │
│                                                              │
│   Terminal Rendering (sveltui)    Metrics Logging           │
│   - Only renders changed cells    - Only logs when needed   │
│   - Differential output           - $effect based           │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                     REACTIVE LAYER                           │
│                                                              │
│   $state(voltage)     $derived(firing)    $effect(log)      │
│        ↓                    ↓                  ↓             │
│   When voltage      Only recomputes      Only runs when     │
│   changes at        for changed          firing changes     │
│   index i...        voltages...                             │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                      COMPUTE LAYER                           │
│                                                              │
│   node-mlx (GPU)                                            │
│   - mx.array for GPU-resident data                          │
│   - .at(indices).add() for scatter operations               │
│   - mx.where() for conditional selection                    │
│   - Async evaluation for non-blocking compute               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Parallel Arrays Pattern

All neuron data stored in parallel arrays. Same index = same neuron.

```typescript
// Neuron state (GPU arrays via node-mlx)
let voltage = $state(mx.array(Float32Array.from({length: N}, () => -70)));
let recovery = $state(mx.array(Float32Array.from({length: N}, () => -14)));
let current = $state(mx.zeros([N]));

// Neuron parameters (could vary per neuron)
const a = mx.full([N], 0.02);  // Recovery time constant
const b = mx.full([N], 0.2);   // Recovery sensitivity
const c = mx.full([N], -65);   // Reset voltage
const d = mx.full([N], 8);     // Reset recovery boost
```

### Synapse Connectivity

Synapses stored as three parallel arrays:

```typescript
// S = total number of synapses
const pre_indices = mx.array([...], mx.int32);   // Shape: [S]
const post_indices = mx.array([...], mx.int32);  // Shape: [S]
let weights = $state(mx.array([...]));           // Shape: [S], learnable
```

For C. elegans scale (302 neurons, ~7000 synapses), this is trivial.
For human scale (86B neurons, ~100T synapses), we'd need hierarchical organization.

---

## The Simulation Step

### 1. Detect Firing

```typescript
// Derived - only recomputes when voltage changes
const firing = $derived(mx.greater(voltage, mx.array(30)));
```

### 2. Transmit Spikes (Sparse!)

```typescript
function transmitSpikes(firing: mx.array, weights: mx.array): mx.array {
    // Get which synapses have a firing pre-neuron
    const synapse_active = firing.index(pre_indices);

    // Compute contribution (0 for inactive synapses)
    const contribution = mx.where(synapse_active, weights, mx.array(0));

    // Scatter-add to post-synaptic neurons
    // This is the KEY operation - only active synapses contribute!
    const post_current = mx.zeros([N]);
    return post_current.at(post_indices).add(contribution);
}
```

### 3. Update Dynamics (Izhikevich)

```typescript
function step(dt: number = 1.0) {
    // Izhikevich equations (all neurons in parallel on GPU)
    const dv = mx.add(
        mx.multiply(0.04, mx.square(voltage)),
        mx.add(
            mx.multiply(5, voltage),
            mx.add(140, mx.subtract(current, recovery))
        )
    );

    const du = mx.multiply(a, mx.subtract(mx.multiply(b, voltage), recovery));

    // Euler integration
    voltage = mx.add(voltage, mx.multiply(dv, dt));
    recovery = mx.add(recovery, mx.multiply(du, dt));

    // Reset firing neurons
    const fired = mx.greater(voltage, mx.array(30));
    voltage = mx.where(fired, c, voltage);
    recovery = mx.where(fired, mx.add(recovery, d), recovery);

    // Clear current for next step
    current = mx.zeros([N]);
}
```

### 4. The Main Loop

```typescript
// One loop is fine - sparse propagation happens WITHIN each step
async function simulate(steps: number) {
    for (let t = 0; t < steps; t++) {
        // Sparse transmission
        const synaptic_current = transmitSpikes(firing, weights);
        current = mx.add(current, synaptic_current);

        // Add external input if any
        current = mx.add(current, external_input);

        // Update dynamics
        step(1.0);

        // Evaluate (actually run GPU computation)
        await mx.asyncEval(voltage, recovery);

        // Reactivity handles the rest:
        // - firing $derived recomputes (only changed neurons)
        // - $effects run (logging, visualization)
        // - sveltui renders (only changed cells)
    }
}
```

---

## STDP Learning

### Eligibility Traces

```typescript
let pre_trace = $state(mx.zeros([N]));   // Recent pre-synaptic activity
let post_trace = $state(mx.zeros([N]));  // Recent post-synaptic activity

function updateTraces(firing: mx.array, dt: number) {
    // Decay
    pre_trace = mx.multiply(pre_trace, mx.exp(mx.array(-dt / tau_plus)));
    post_trace = mx.multiply(post_trace, mx.exp(mx.array(-dt / tau_minus)));

    // Increment on spike
    pre_trace = mx.where(firing, mx.add(pre_trace, mx.array(1)), pre_trace);
    post_trace = mx.where(firing, mx.add(post_trace, mx.array(1)), post_trace);
}
```

### Weight Updates

```typescript
function applySTDP(pre_firing: mx.array, post_firing: mx.array) {
    // Get traces at synapse locations
    const pre_trace_at_syn = pre_trace.index(pre_indices);
    const post_trace_at_syn = post_trace.index(post_indices);

    // LTP: post fires, pre was recently active
    const post_fired_at_syn = post_firing.index(post_indices);
    const ltp = mx.where(post_fired_at_syn,
                         mx.multiply(A_plus, pre_trace_at_syn),
                         mx.array(0));

    // LTD: pre fires, post was recently active
    const pre_fired_at_syn = pre_firing.index(pre_indices);
    const ltd = mx.where(pre_fired_at_syn,
                         mx.multiply(mx.array(-A_minus), post_trace_at_syn),
                         mx.array(0));

    // Update weights
    weights = mx.clip(mx.add(weights, mx.add(ltp, ltd)),
                      mx.array(0), mx.array(1));
}
```

### Reward Modulation

```typescript
function applyReward(reward: number) {
    if (reward === 0) return;

    // Modulate recent STDP based on reward
    // Positive reward: amplify recent changes
    // Negative reward: reverse recent changes
    const modulation = reward > 0 ? dopamine_boost : -punishment_factor;

    // Apply to eligibility traces or directly to recent weight changes
    // (Implementation depends on exact biological model we choose)
}
```

---

## Visualization with sveltui

### World Rendering

```svelte
<!-- WorldView.svelte -->
<script lang="ts">
    import { Box, Text } from 'sveltui';
    import { world, creature } from '../world/world.svelte.ts';

    // Reactive - only updates when world changes
    const grid = $derived(renderGrid(world, creature));
</script>

<Box border="single" width={world.width + 2} height={world.height + 2}>
    {#each grid as row, y}
        <Box flexDirection="row">
            {#each row as cell, x}
                <Text
                    text={cell.char}
                    color={cell.color}
                />
            {/each}
        </Box>
    {/each}
</Box>
```

### Metrics Panel

```svelte
<!-- Metrics.svelte -->
<script lang="ts">
    import { Box, Text } from 'sveltui';
    import { network } from '../core/network.svelte.ts';

    // Only recomputes when network state changes
    const spikeRate = $derived(computeSpikeRate(network.firing));
    const avgWeight = $derived(computeAvgWeight(network.weights));
</script>

<Box flexDirection="column" padding={1}>
    <Text text={`Neurons: ${network.n}`} />
    <Text text={`Spike Rate: ${spikeRate.toFixed(1)}/s`} />
    <Text text={`Avg Weight: ${avgWeight.toFixed(3)}`} />
</Box>
```

---

## Scaling Considerations

### Memory Budget (64GB M1 Max)

| Component | Per Neuron | 302 Neurons | 1M Neurons | 1B Neurons |
|-----------|------------|-------------|------------|------------|
| Voltage (f32) | 4 bytes | 1.2 KB | 4 MB | 4 GB |
| Recovery (f32) | 4 bytes | 1.2 KB | 4 MB | 4 GB |
| Parameters | 16 bytes | 4.8 KB | 16 MB | 16 GB |
| **Total State** | 24 bytes | 7.2 KB | 24 MB | 24 GB |

Synapses scale differently:
| Connectivity | Synapses | Memory (12 bytes each) |
|--------------|----------|------------------------|
| C. elegans | ~7,000 | 84 KB |
| 1M neurons, 1% | 10M | 120 MB |
| 1B neurons, 0.1% | 1B | 12 GB |

**With 64GB: ~1-2 billion neurons theoretically possible for storage.**

### Compute Budget

The key is SPARSE computation:
- If only 1% of neurons fire each step
- And each neuron has ~1000 synapses
- 1B neurons × 1% × 1000 = 10B operations per step
- M1 Max GPU: ~10 TFLOPS = 1000 steps/second at this scale

**This is why sparse matters. Dense would be impossible.**

---

## File Organization

```
src/
├── core/
│   ├── neuron.svelte.ts      # NeuronPopulation class
│   │   ├── $state: voltage, recovery, current
│   │   ├── $derived: firing
│   │   ├── step(): Izhikevich dynamics
│   │   └── reset(): Clear state
│   │
│   ├── synapse.svelte.ts     # SynapseGroup class
│   │   ├── pre_indices, post_indices (fixed)
│   │   ├── $state: weights
│   │   ├── transmit(): Sparse spike transmission
│   │   └── applySTDP(): Learning
│   │
│   └── network.svelte.ts     # Network orchestration
│       ├── populations: NeuronPopulation[]
│       ├── connections: SynapseGroup[]
│       └── step(): Coordinate simulation
│
├── world/
│   ├── world.svelte.ts       # 2D environment
│   │   ├── $state: grid, food_positions, danger_positions
│   │   ├── $derived: sensory_input
│   │   └── move(), reset()
│   │
│   └── creature.svelte.ts    # Brain-body interface
│       ├── brain: Network
│       ├── position: [x, y]
│       └── step(): Sense → Think → Act
│
├── components/               # sveltui visualization
│   ├── WorldView.svelte
│   ├── BrainView.svelte
│   └── Metrics.svelte
│
└── main.ts                   # Entry point
```

---

## Testing Strategy

### Unit Tests

```typescript
// Test individual building blocks
describe('NeuronPopulation', () => {
    it('should fire when voltage exceeds threshold', () => {
        const pop = new NeuronPopulation(10);
        pop.voltage.indexPut_([0], 35);  // Above threshold
        expect(mx.sum(pop.firing).item()).toBe(1);
    });
});
```

### Integration Tests

```typescript
// Test sparse propagation
describe('SynapseGroup', () => {
    it('should only transmit from firing neurons', () => {
        // Create 10 neurons, only neuron 0 fires
        // Verify only synapses FROM neuron 0 contribute
    });
});
```

### Scaling Tests

```typescript
// Verify O(k) not O(n) performance
describe('Scaling', () => {
    it('should scale with active neurons, not total', () => {
        // 1000 neurons, 10 firing vs 100 firing
        // Time should scale ~10x, not stay constant
    });
});
```

### Learning Tests (The Slap)

```typescript
// CRITICAL: Don't fool ourselves
describe('Learning', () => {
    it('should improve over random baseline', () => {
        // Run with learning
        // Run with random weights
        // Learning should significantly outperform
    });

    it('should not work with shuffled rewards', () => {
        // If learning works with random rewards, it's not learning
    });
});
```

---

## Next Steps

1. **Create project**: `bunx sveltui create brain`
2. **Add node-mlx**: `bun add @frost-beta/mlx`
3. **Implement NeuronPopulation** with Svelte reactivity
4. **Implement SynapseGroup** with scatter-add
5. **Create simple test**: 100 neurons, random connectivity, verify spikes propagate
6. **Add visualization**: Watch neurons fire in terminal
7. **Build creature**: Sensory → Brain → Motor
8. **Test learning**: Does STDP + reward actually work?

---

*The Vessel Architecture*
*December 2025*
