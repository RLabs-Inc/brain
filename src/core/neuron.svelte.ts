/**
 * NeuronPopulation - Reactive Izhikevich Neuron Building Block
 *
 * Following sveltui pattern EXACTLY:
 * - Direct $state exports for all arrays
 * - SvelteMap for reactive registry
 * - Free index pool for reuse
 * - Double function pattern ONLY for derived values
 * - ALL computation on GPU - NEVER convert to JS
 * - Direct mutation of state arrays
 *
 * Biological fidelity:
 * - Dale's Law: neurons are excitatory OR inhibitory, never both
 * - Correct Izhikevich parameters from original paper
 * - Background noise (~5) for spontaneous activity
 * - Neuron roles and regions for DNA/genome system
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import { SvelteMap } from 'svelte/reactivity'

// ============================================================================
// NEURON TYPE PRESETS (Izhikevich parameters from original 2003 paper)
// ============================================================================

// Excitatory neuron types (typically glutamatergic)
export const ExcitatoryTypes = {
  RS: { a: 0.02, b: 0.2, c: -65, d: 8 },     // Regular spiking - most common cortical
  IB: { a: 0.02, b: 0.2, c: -55, d: 4 },     // Intrinsically bursting - layer 5
  CH: { a: 0.02, b: 0.2, c: -50, d: 2 },     // Chattering - layer 4
  TC: { a: 0.02, b: 0.25, c: -65, d: 0.05 }, // Thalamo-cortical
  RZ: { a: 0.1, b: 0.26, c: -65, d: 2 },     // Resonator
} as const

// Inhibitory neuron types (typically GABAergic)
export const InhibitoryTypes = {
  FS: { a: 0.1, b: 0.2, c: -65, d: 2 },      // Fast spiking - basket cells
  LTS: { a: 0.02, b: 0.25, c: -65, d: 2 },   // Low-threshold spiking - Martinotti
} as const

// Combined for backward compatibility
export const NeuronTypes = {
  ...ExcitatoryTypes,
  ...InhibitoryTypes,
} as const

// Firing threshold constant (mV)
export const THRESHOLD = 30

// ============================================================================
// BIOLOGICAL CONSTANTS (from Izhikevich 2003)
// ============================================================================

// Default noise amplitude for thalamic/background input
export const DEFAULT_NOISE_AMPLITUDE = 5.0

// Weight scale guidelines (for reference - enforced in synapse module)
// Excitatory synapses: 0 to 0.5
// Inhibitory synapses: -1 to 0

// ============================================================================
// NEURON ROLES (for DNA/genome system)
// ============================================================================

export type NeuronRole = 'sensory' | 'motor' | 'inter' | 'modulatory'
export type BrainRegion = 'cortex' | 'thalamus' | 'basal_ganglia' | 'cerebellum' | 'brainstem' | 'other'

// ============================================================================
// POPULATION REGISTRY (like sveltui engine.svelte.ts registry)
// ============================================================================

export const registry = $state({
  idToIndex: new SvelteMap<string, number>(),
  indexToId: new SvelteMap<number, string>(),
  allocatedIndices: new Set<number>(),
  freeIndices: [] as number[],
  nextIndex: 0,
})

// ============================================================================
// POPULATION STATE (DIRECT EXPORTS)
// ============================================================================

// Population metadata
export const populationSize = $state<number[]>([])
export const populationType = $state<string[]>([])

// Dale's Law: each population is either excitatory or inhibitory
export const isExcitatory = $state<boolean[]>([])

// DNA/Genome metadata
export const neuronRole = $state<NeuronRole[]>([])
export const neuronRegion = $state<BrainRegion[]>([])

// Neuron state arrays - GPU arrays indexed by [populationIndex]
export const voltage = $state<ReturnType<typeof mx.array>[]>([])
export const recovery = $state<ReturnType<typeof mx.array>[]>([])
export const current = $state<ReturnType<typeof mx.array>[]>([])
export const fired = $state<ReturnType<typeof mx.array>[]>([])  // Which neurons fired THIS timestep

// Izhikevich parameters per population (GPU arrays)
export const paramA = $state<ReturnType<typeof mx.array>[]>([])
export const paramB = $state<ReturnType<typeof mx.array>[]>([])
export const paramC = $state<ReturnType<typeof mx.array>[]>([])
export const paramD = $state<ReturnType<typeof mx.array>[]>([])

// Background noise for spontaneous activity (GPU scalars per population)
// ~5 is biologically realistic for thalamic input (Izhikevich 2003)
export const noiseAmplitude = $state<ReturnType<typeof mx.array>[]>([])

// ============================================================================
// POPULATION MANAGEMENT (like sveltui allocateIndex/releaseIndex)
// ============================================================================

/**
 * Options for allocating a neuron population.
 */
export interface PopulationOptions {
  type?: keyof typeof NeuronTypes
  excitatory?: boolean  // Override auto-detection from type
  role?: NeuronRole
  region?: BrainRegion
  noise?: number  // Override default noise amplitude
}

/**
 * Allocate a new neuron population.
 * Returns the population index.
 *
 * Dale's Law: Neuron type determines if excitatory or inhibitory.
 * - RS, IB, CH, TC, RZ → excitatory (glutamatergic)
 * - FS, LTS → inhibitory (GABAergic)
 *
 * @param id - Unique identifier for the population
 * @param size - Number of neurons (1 for individual neuron)
 * @param options - Configuration options
 */
export function allocatePopulation(
  id: string,
  size: number,
  options: PopulationOptions = {}
): number {
  // Check if already exists
  const existing = registry.idToIndex.get(id)
  if (existing !== undefined) return existing

  // Reuse freed index or allocate new
  let index: number
  if (registry.freeIndices.length > 0) {
    index = registry.freeIndices.pop()!
  } else {
    index = registry.nextIndex++
  }

  // Update registry
  registry.idToIndex.set(id, index)
  registry.indexToId.set(index, id)
  registry.allocatedIndices.add(index)

  const type = options.type ?? 'RS'
  const preset = NeuronTypes[type]

  // Dale's Law: auto-detect excitatory/inhibitory from type unless overridden
  const excitatory = options.excitatory ?? (type in ExcitatoryTypes)

  // Initialize metadata
  populationSize[index] = size
  populationType[index] = type
  isExcitatory[index] = excitatory
  neuronRole[index] = options.role ?? 'inter'
  neuronRegion[index] = options.region ?? 'other'

  // Initialize neuron state (GPU arrays)
  voltage[index] = mx.full([size], -70, mx.float32)
  recovery[index] = mx.full([size], -14, mx.float32)
  current[index] = mx.zeros([size], mx.float32)
  fired[index] = mx.zeros([size], mx.bool_)  // No neurons firing initially

  // Initialize Izhikevich parameters (GPU arrays)
  paramA[index] = mx.full([size], preset.a, mx.float32)
  paramB[index] = mx.full([size], preset.b, mx.float32)
  paramC[index] = mx.full([size], preset.c, mx.float32)
  paramD[index] = mx.full([size], preset.d, mx.float32)

  // Background noise amplitude (GPU scalar)
  noiseAmplitude[index] = mx.array(options.noise ?? DEFAULT_NOISE_AMPLITUDE, mx.float32)

  return index
}

/**
 * Get population index by id.
 */
export function getPopulationIndex(id: string): number | undefined {
  return registry.idToIndex.get(id)
}

/**
 * Release a population (like sveltui releaseIndex).
 */
export function releasePopulation(id: string): void {
  const index = registry.idToIndex.get(id)
  if (index === undefined) return

  // Update registry
  registry.idToIndex.delete(id)
  registry.indexToId.delete(index)
  registry.allocatedIndices.delete(index)
  registry.freeIndices.push(index)

  // Clean up arrays (set to empty/default values)
  populationSize[index] = 0
  populationType[index] = ''
  isExcitatory[index] = true
  neuronRole[index] = 'inter'
  neuronRegion[index] = 'other'
  voltage[index] = mx.zeros([0], mx.float32)
  recovery[index] = mx.zeros([0], mx.float32)
  current[index] = mx.zeros([0], mx.float32)
  fired[index] = mx.zeros([0], mx.bool_)
  paramA[index] = mx.zeros([0], mx.float32)
  paramB[index] = mx.zeros([0], mx.float32)
  paramC[index] = mx.zeros([0], mx.float32)
  paramD[index] = mx.zeros([0], mx.float32)
  noiseAmplitude[index] = mx.array(0, mx.float32)
}

// ============================================================================
// NEURON OPERATIONS (DIRECT EXPORTS - operate on GPU arrays)
// ============================================================================

/**
 * Inject current into neurons via scatter-add.
 * Multiple injections to same neuron accumulate.
 *
 * @param popIndex - Population index
 * @param indices - Which neurons (GPU array, int32)
 * @param values - How much current (GPU array, float32)
 */
export function injectCurrent(
  popIndex: number,
  indices: ReturnType<typeof mx.array>,
  values: ReturnType<typeof mx.array>
) {
  current[popIndex] = current[popIndex].at(indices).add(values)
}

/**
 * Inject uniform current to all neurons in population.
 */
export function injectUniformCurrent(popIndex: number, value: number) {
  current[popIndex] = mx.add(current[popIndex], mx.array(value))
}

/**
 * Integrate Izhikevich equations for one timestep with sub-stepping.
 * Uses 4 sub-steps of 0.25ms each for numerical stability.
 * ALL computation on GPU - no CPU loops.
 *
 * Includes background noise injection (~5 Gaussian) for spontaneous activity.
 * This mimics thalamic input that keeps neurons near threshold.
 *
 * @param popIndex - Population index
 * @param dt - Total timestep in ms (default 1.0)
 * @param injectNoise - Whether to inject background noise (default true)
 */
export function integrate(popIndex: number, dt: number = 1.0, injectNoise: boolean = true) {
  let v = voltage[popIndex]
  let u = recovery[popIndex]
  const size = populationSize[popIndex]

  // Get synaptic current + optional background noise (ALL GPU)
  let I = current[popIndex]
  if (injectNoise) {
    // Gaussian noise with amplitude ~5 (biologically realistic thalamic input)
    const noise = mx.multiply(
      noiseAmplitude[popIndex],
      mx.random.normal([size])
    )
    I = mx.add(I, noise)
  }

  const a = paramA[popIndex]
  const b = paramB[popIndex]
  const c = paramC[popIndex]
  const d = paramD[popIndex]

  // Track any spikes across all sub-steps
  let anyFired = mx.zeros([size], mx.bool_)

  // Sub-stepping for numerical stability (4 steps of 0.25ms)
  const subDt = dt / 4
  for (let i = 0; i < 4; i++) {
    // Izhikevich equations
    // dv/dt = 0.04v² + 5v + 140 - u + I
    const dv = mx.add(
      mx.multiply(0.04, mx.square(v)),
      mx.add(mx.multiply(5, v), mx.add(140, mx.subtract(I, u)))
    )

    // du/dt = a(bv - u)
    const du = mx.multiply(a, mx.subtract(mx.multiply(b, v), u))

    // Euler integration for this sub-step
    v = mx.add(v, mx.multiply(dv, subDt))
    u = mx.add(u, mx.multiply(du, subDt))

    // Check for spikes within sub-step and reset
    const spikeMask = mx.greaterEqual(v, mx.array(THRESHOLD))
    anyFired = mx.logicalOr(anyFired, spikeMask)  // Track all spikes
    v = mx.where(spikeMask, c, v)
    u = mx.where(spikeMask, mx.add(u, d), u)
  }

  // Store final values
  voltage[popIndex] = v
  recovery[popIndex] = u
  fired[popIndex] = anyFired  // Store which neurons fired this timestep

  // Clear current for next timestep
  current[popIndex] = mx.zeros([size], mx.float32)
}

/**
 * Reset population to resting state.
 */
export function resetPopulation(popIndex: number) {
  const size = populationSize[popIndex]
  voltage[popIndex] = mx.full([size], -70, mx.float32)
  recovery[popIndex] = mx.full([size], -14, mx.float32)
  current[popIndex] = mx.zeros([size], mx.float32)
  fired[popIndex] = mx.zeros([size], mx.bool_)
}

/**
 * Set noise amplitude for a population.
 * ~5 is biologically realistic (thalamic input).
 * 0 disables noise entirely.
 */
export function setNoiseAmplitude(popIndex: number, amplitude: number) {
  noiseAmplitude[popIndex] = mx.array(amplitude, mx.float32)
}

/**
 * Check if a neuron type is excitatory.
 * Used for Dale's Law enforcement.
 */
export function isTypeExcitatory(type: keyof typeof NeuronTypes): boolean {
  return type in ExcitatoryTypes
}

/**
 * Get the appropriate weight bounds for a pre-synaptic population.
 * Excitatory: 0 to 0.5 (Izhikevich paper)
 * Inhibitory: -1 to 0 (Izhikevich paper)
 */
export function getWeightBounds(prePopIndex: number): { min: number; max: number } {
  if (isExcitatory[prePopIndex]) {
    return { min: 0, max: 0.5 }
  } else {
    return { min: -1, max: 0 }
  }
}

// ============================================================================
// DERIVED VALUES - Double function pattern (like sveltui getEngine)
// ALL values stay on GPU - no .item() or .tolist() ever!
// ============================================================================

/**
 * Get derived values for a population.
 * Uses double function pattern for reactive derivations.
 * ALL values are GPU arrays - caller must use mx operations to read them.
 */
export function getNeuronDerived(popIndex: number) {
  // Which neurons fired THIS timestep (from integrate)
  const firing = $derived(fired[popIndex])

  // Total spike count - GPU scalar (use array.item() only at final output!)
  const spikeCount = $derived(mx.sum(fired[popIndex]))

  // Average voltage - GPU scalar
  const avgVoltage = $derived(mx.mean(voltage[popIndex]))

  // Return through double function - ALL values are GPU arrays/scalars
  return () => ({
    firing,      // GPU boolean array
    spikeCount,  // GPU scalar
    avgVoltage,  // GPU scalar
  })
}
