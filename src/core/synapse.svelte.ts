/**
 * SynapseGroup - Reactive Sparse Synaptic Transmission Building Block
 *
 * Following sveltui pattern EXACTLY:
 * - Direct $state exports for all arrays
 * - SvelteMap for reactive registry
 * - Free index pool for reuse
 * - Double function pattern ONLY for derived values
 * - ALL computation on GPU - NEVER convert to JS
 * - Direct mutation of state arrays
 *
 * KEY OPERATION: scatter-add for sparse transmission
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import { SvelteMap } from 'svelte/reactivity'
import {
  voltage,
  current,
  fired,
  populationSize,
  THRESHOLD,
  registry as neuronRegistry,
  isExcitatory,
  getWeightBounds,
} from './neuron.svelte.ts'

// ============================================================================
// SYNAPSE TYPES (for more biologically accurate transmission)
// ============================================================================

export type SynapseType = 'AMPA' | 'NMDA' | 'GABA_A' | 'GABA_B'

// Synapse type defaults (time constants in ms)
export const SynapseTypeDefaults = {
  AMPA: { tauPlus: 20, tauMinus: 20, aPlus: 0.01, aMinus: 0.012 },   // Fast excitatory
  NMDA: { tauPlus: 50, tauMinus: 50, aPlus: 0.005, aMinus: 0.006 },  // Slow excitatory
  GABA_A: { tauPlus: 20, tauMinus: 20, aPlus: 0.01, aMinus: 0.012 }, // Fast inhibitory
  GABA_B: { tauPlus: 100, tauMinus: 100, aPlus: 0.002, aMinus: 0.003 }, // Slow inhibitory
} as const

// ============================================================================
// SYNAPSE GROUP REGISTRY (like sveltui engine.svelte.ts registry)
// ============================================================================

export const registry = $state({
  idToIndex: new SvelteMap<string, number>(),
  indexToId: new SvelteMap<number, string>(),
  allocatedIndices: new Set<number>(),
  freeIndices: [] as number[],
  nextIndex: 0,
})

// ============================================================================
// SYNAPSE GROUP STATE (DIRECT EXPORTS)
// ============================================================================

// Group metadata
export const groupSynapseCount = $state<number[]>([])
export const groupPrePopIndex = $state<number[]>([])   // Which population synapses come FROM
export const groupPostPopIndex = $state<number[]>([])  // Which population synapses go TO
export const groupSynapseType = $state<SynapseType[]>([])  // AMPA, NMDA, GABA_A, GABA_B
export const groupIsPlastic = $state<boolean[]>([])    // Whether STDP is enabled

// Connectivity - GPU arrays (parallel arrays: same index = same synapse)
export const preIndices = $state<ReturnType<typeof mx.array>[]>([])   // Pre-neuron index for each synapse
export const postIndices = $state<ReturnType<typeof mx.array>[]>([])  // Post-neuron index for each synapse

// Weights - GPU arrays (learnable)
export const weights = $state<ReturnType<typeof mx.array>[]>([])

// Weight bounds (GPU scalars for GPU operations)
// AUTOMATICALLY SET FROM DALE'S LAW:
// Excitatory pre-neuron: 0 to 0.5
// Inhibitory pre-neuron: -1 to 0
export const minWeight = $state<ReturnType<typeof mx.array>[]>([])
export const maxWeight = $state<ReturnType<typeof mx.array>[]>([])

// STDP traces - GPU arrays
export const preTrace = $state<ReturnType<typeof mx.array>[]>([])   // Per pre-neuron
export const postTrace = $state<ReturnType<typeof mx.array>[]>([])  // Per post-neuron

// Eligibility trace per synapse (for three-factor learning)
export const eligibility = $state<ReturnType<typeof mx.array>[]>([])

// STDP parameters per group (GPU scalars for GPU operations)
export const stdpTauPlus = $state<ReturnType<typeof mx.array>[]>([])
export const stdpTauMinus = $state<ReturnType<typeof mx.array>[]>([])
export const stdpAPlus = $state<ReturnType<typeof mx.array>[]>([])
export const stdpAMinus = $state<ReturnType<typeof mx.array>[]>([])

// Pre-computed decay factors (GPU scalars)
export const decayPlus = $state<ReturnType<typeof mx.array>[]>([])
export const decayMinus = $state<ReturnType<typeof mx.array>[]>([])

// ============================================================================
// SYNAPSE GROUP MANAGEMENT (like sveltui allocateIndex/releaseIndex)
// ============================================================================

/**
 * Options for allocating a synapse group.
 */
export interface SynapseGroupOptions {
  initialWeights?: ReturnType<typeof mx.array>  // GPU array!
  synapseType?: SynapseType  // Override auto-detection
  plastic?: boolean  // Whether STDP is enabled (default true)
  // These override Dale's Law auto-detection (use with caution!)
  minWeight?: number
  maxWeight?: number
  // STDP parameters (override synapse type defaults)
  tauPlus?: number
  tauMinus?: number
  aPlus?: number
  aMinus?: number
}

/**
 * Allocate a new synapse group.
 * Connectivity is passed as GPU arrays - no JS array conversion!
 *
 * DALE'S LAW ENFORCEMENT:
 * - Weight bounds are automatically set based on pre-synaptic neuron type
 * - Excitatory pre-neurons: weights 0 to 0.5
 * - Inhibitory pre-neurons: weights -1 to 0
 * - Synapse type auto-detected: excitatory → AMPA, inhibitory → GABA_A
 *
 * @param id - Unique identifier
 * @param prePopIndex - Source population index
 * @param postPopIndex - Target population index
 * @param preIndicesGPU - Which pre-neurons connect (GPU array)
 * @param postIndicesGPU - Which post-neurons they connect to (GPU array)
 * @param options - Optional configuration
 */
export function allocateSynapseGroup(
  id: string,
  prePopIndex: number,
  postPopIndex: number,
  preIndicesGPU: ReturnType<typeof mx.array>,  // GPU array!
  postIndicesGPU: ReturnType<typeof mx.array>, // GPU array!
  options: SynapseGroupOptions = {}
): number {
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

  // Get sizes from GPU array shape (stays on GPU)
  const synapseCount = preIndicesGPU.shape[0]
  const preSize = populationSize[prePopIndex]
  const postSize = populationSize[postPopIndex]

  // DALE'S LAW: Auto-detect synapse type and weight bounds from pre-neuron type
  const preIsExcitatory = isExcitatory[prePopIndex]
  const autoSynapseType: SynapseType = preIsExcitatory ? 'AMPA' : 'GABA_A'
  const daleBounds = getWeightBounds(prePopIndex)

  // Metadata
  groupSynapseCount[index] = synapseCount
  groupPrePopIndex[index] = prePopIndex
  groupPostPopIndex[index] = postPopIndex
  groupSynapseType[index] = options.synapseType ?? autoSynapseType
  groupIsPlastic[index] = options.plastic ?? true

  // Connectivity (already GPU arrays)
  preIndices[index] = preIndicesGPU
  postIndices[index] = postIndicesGPU

  // Weight bounds from Dale's Law (can be overridden with caution)
  const minW = options.minWeight ?? daleBounds.min
  const maxW = options.maxWeight ?? daleBounds.max
  minWeight[index] = mx.array(minW, mx.float32)
  maxWeight[index] = mx.array(maxW, mx.float32)

  // Weights (GPU array)
  if (options.initialWeights) {
    weights[index] = options.initialWeights
  } else {
    // Random initialization between min and max - ALL ON GPU
    weights[index] = mx.add(
      minWeight[index],
      mx.multiply(
        mx.random.uniform(0, 1, [synapseCount]),
        mx.subtract(maxWeight[index], minWeight[index])
      )
    )
  }

  // STDP parameters from synapse type defaults (can be overridden)
  const synType = groupSynapseType[index]
  const typeDefaults = SynapseTypeDefaults[synType]
  const tauP = options.tauPlus ?? typeDefaults.tauPlus
  const tauM = options.tauMinus ?? typeDefaults.tauMinus
  stdpTauPlus[index] = mx.array(tauP, mx.float32)
  stdpTauMinus[index] = mx.array(tauM, mx.float32)
  stdpAPlus[index] = mx.array(options.aPlus ?? typeDefaults.aPlus, mx.float32)
  stdpAMinus[index] = mx.array(options.aMinus ?? typeDefaults.aMinus, mx.float32)

  // Pre-compute decay factors (GPU scalars)
  decayPlus[index] = mx.array(Math.exp(-1 / tauP), mx.float32)
  decayMinus[index] = mx.array(Math.exp(-1 / tauM), mx.float32)

  // STDP traces (GPU arrays)
  preTrace[index] = mx.zeros([preSize], mx.float32)
  postTrace[index] = mx.zeros([postSize], mx.float32)

  // Eligibility trace (GPU array)
  eligibility[index] = mx.zeros([synapseCount], mx.float32)

  return index
}

/**
 * Get synapse group index by id.
 */
export function getSynapseGroupIndex(id: string): number | undefined {
  return registry.idToIndex.get(id)
}

/**
 * Release a synapse group.
 */
export function releaseSynapseGroup(id: string): void {
  const index = registry.idToIndex.get(id)
  if (index === undefined) return

  // Update registry
  registry.idToIndex.delete(id)
  registry.indexToId.delete(index)
  registry.allocatedIndices.delete(index)
  registry.freeIndices.push(index)

  // Clean up (empty GPU arrays / default values)
  groupSynapseCount[index] = 0
  groupPrePopIndex[index] = -1
  groupPostPopIndex[index] = -1
  groupSynapseType[index] = 'AMPA'
  groupIsPlastic[index] = true
  preIndices[index] = mx.zeros([0], mx.int32)
  postIndices[index] = mx.zeros([0], mx.int32)
  weights[index] = mx.zeros([0], mx.float32)
  minWeight[index] = mx.array(0, mx.float32)
  maxWeight[index] = mx.array(0.5, mx.float32)
  preTrace[index] = mx.zeros([0], mx.float32)
  postTrace[index] = mx.zeros([0], mx.float32)
  eligibility[index] = mx.zeros([0], mx.float32)
  stdpTauPlus[index] = mx.array(20, mx.float32)
  stdpTauMinus[index] = mx.array(20, mx.float32)
  stdpAPlus[index] = mx.array(0.01, mx.float32)
  stdpAMinus[index] = mx.array(0.012, mx.float32)
  decayPlus[index] = mx.array(0.95, mx.float32)
  decayMinus[index] = mx.array(0.95, mx.float32)
}

// ============================================================================
// SYNAPSE OPERATIONS (ALL GPU - no CPU loops!)
// ============================================================================

/**
 * Transmit spikes from pre to post population.
 * THIS IS THE KEY SPARSE OPERATION!
 * ALL on GPU - no CPU loops.
 *
 * NOTE: We copy postIndices before scatter-add due to node-mlx bug where
 * .at().add() corrupts the index array for subsequent .index() operations.
 */
export function transmit(groupIndex: number) {
  const prePopIndex = groupPrePopIndex[groupIndex]
  const postPopIndex = groupPostPopIndex[groupIndex]

  // Use the fired state from last integration (not voltage comparison)
  const preFiring = fired[prePopIndex]

  // Get which synapses have a firing pre-neuron (GPU indexing)
  const synapseActive = preFiring.index(preIndices[groupIndex])

  // Compute contribution (weight if active, 0 otherwise) - GPU where
  const contribution = mx.where(
    synapseActive,
    weights[groupIndex],
    mx.zerosLike(weights[groupIndex])
  )

  // Copy postIndices to avoid corruption from scatter-add (node-mlx bug workaround)
  const postIdxCopy = mx.array(postIndices[groupIndex])

  // Scatter-add to post-synaptic neurons - THE KEY OPERATION!
  current[postPopIndex] = current[postPopIndex].at(postIdxCopy).add(contribution)
}

/**
 * Update STDP traces based on current firing.
 * ALL GPU operations.
 * Only updates if group is plastic.
 */
export function updateTraces(groupIndex: number) {
  // Skip if not plastic
  if (!groupIsPlastic[groupIndex]) return

  const prePopIndex = groupPrePopIndex[groupIndex]
  const postPopIndex = groupPostPopIndex[groupIndex]

  // Use the fired state from last integration
  const preFiring = fired[prePopIndex]
  const postFiring = fired[postPopIndex]

  // Decay traces (GPU multiply with pre-computed decay factors)
  preTrace[groupIndex] = mx.multiply(preTrace[groupIndex], decayPlus[groupIndex])
  postTrace[groupIndex] = mx.multiply(postTrace[groupIndex], decayMinus[groupIndex])

  // Increment traces for firing neurons (GPU where)
  preTrace[groupIndex] = mx.where(
    preFiring,
    mx.add(preTrace[groupIndex], mx.array(1.0)),
    preTrace[groupIndex]
  )
  postTrace[groupIndex] = mx.where(
    postFiring,
    mx.add(postTrace[groupIndex], mx.array(1.0)),
    postTrace[groupIndex]
  )
}

/**
 * Apply STDP learning rule.
 * Updates eligibility trace and optionally weights directly.
 * ALL GPU operations.
 * Only applies if group is plastic.
 */
export function applySTDP(groupIndex: number, directUpdate: boolean = false) {
  // Skip if not plastic
  if (!groupIsPlastic[groupIndex]) return

  const prePopIndex = groupPrePopIndex[groupIndex]
  const postPopIndex = groupPostPopIndex[groupIndex]

  // Use the fired state from last integration
  const preFiring = fired[prePopIndex]
  const postFiring = fired[postPopIndex]

  // Get traces at synapse locations (GPU indexing)
  const preTraceAtSyn = preTrace[groupIndex].index(preIndices[groupIndex])
  const postTraceAtSyn = postTrace[groupIndex].index(postIndices[groupIndex])

  // Get firing at synapse locations (GPU indexing)
  const preFiredAtSyn = preFiring.index(preIndices[groupIndex])
  const postFiredAtSyn = postFiring.index(postIndices[groupIndex])

  // LTP: Post fires AND pre was recently active (GPU where)
  const ltp = mx.where(
    postFiredAtSyn,
    mx.multiply(stdpAPlus[groupIndex], preTraceAtSyn),
    mx.zerosLike(weights[groupIndex])
  )

  // LTD: Pre fires AND post was recently active (GPU where)
  const ltd = mx.where(
    preFiredAtSyn,
    mx.multiply(mx.negative(stdpAMinus[groupIndex]), postTraceAtSyn),
    mx.zerosLike(weights[groupIndex])
  )

  // Update eligibility trace (GPU)
  const dw = mx.add(ltp, ltd)
  eligibility[groupIndex] = mx.add(
    mx.multiply(eligibility[groupIndex], mx.array(0.95)),
    dw
  )

  // Optionally apply directly to weights
  if (directUpdate) {
    weights[groupIndex] = mx.clip(
      mx.add(weights[groupIndex], dw),
      minWeight[groupIndex],
      maxWeight[groupIndex]
    )
  }
}

/**
 * Apply reward-modulated learning (three-factor rule).
 * ALL GPU operations.
 */
export function applyReward(groupIndex: number, reward: ReturnType<typeof mx.array>) {
  // Modulate weight change by reward (GPU multiply)
  const dw = mx.multiply(eligibility[groupIndex], reward)

  // Apply and clip (GPU)
  weights[groupIndex] = mx.clip(
    mx.add(weights[groupIndex], dw),
    minWeight[groupIndex],
    maxWeight[groupIndex]
  )

  // Decay eligibility (GPU)
  eligibility[groupIndex] = mx.multiply(eligibility[groupIndex], mx.array(0.5))
}

/**
 * Reset learning state.
 */
export function resetLearning(groupIndex: number) {
  const preSize = populationSize[groupPrePopIndex[groupIndex]]
  const postSize = populationSize[groupPostPopIndex[groupIndex]]
  const synapseCount = groupSynapseCount[groupIndex]

  preTrace[groupIndex] = mx.zeros([preSize], mx.float32)
  postTrace[groupIndex] = mx.zeros([postSize], mx.float32)
  eligibility[groupIndex] = mx.zeros([synapseCount], mx.float32)
}

/**
 * Reset weights to random.
 */
export function resetWeights(groupIndex: number) {
  const synapseCount = groupSynapseCount[groupIndex]
  weights[groupIndex] = mx.add(
    minWeight[groupIndex],
    mx.multiply(
      mx.random.uniform(0, 1, [synapseCount]),
      mx.subtract(maxWeight[groupIndex], minWeight[groupIndex])
    )
  )
}

// ============================================================================
// DERIVED VALUES - Double function pattern
// ALL values stay on GPU!
// ============================================================================

/**
 * Get derived values for a synapse group.
 * ALL values are GPU arrays/scalars.
 */
export function getSynapseDerived(groupIndex: number) {
  // Average weight - GPU scalar
  const avgWeight = $derived(mx.mean(weights[groupIndex]))

  // Weight statistics - GPU scalars
  const weightMin = $derived(mx.min(weights[groupIndex]))
  const weightMax = $derived(mx.max(weights[groupIndex]))

  // Average eligibility - GPU scalar
  const avgEligibility = $derived(mx.mean(mx.abs(eligibility[groupIndex])))

  return () => ({
    avgWeight,      // GPU scalar
    weightMin,      // GPU scalar
    weightMax,      // GPU scalar
    avgEligibility, // GPU scalar
  })
}

// ============================================================================
// CONNECTIVITY UTILITIES - ALL GPU!
// ============================================================================

/**
 * Create random sparse connectivity - ALL ON GPU.
 * Returns GPU arrays directly.
 */
export function createRandomConnectivity(
  preSize: number,
  postSize: number,
  density: number
): { preIndices: ReturnType<typeof mx.array>; postIndices: ReturnType<typeof mx.array> } {
  // Create all possible connections as GPU arrays
  const totalPossible = preSize * postSize

  // Generate random mask on GPU
  const mask = mx.less(mx.random.uniform(0, 1, [totalPossible]), mx.array(density))

  // Create index grids on GPU
  const preGrid = mx.floor(mx.divide(mx.arange(totalPossible), mx.array(postSize)))
  const postGrid = mx.remainder(mx.arange(totalPossible), mx.array(postSize))

  // Select only connected indices (GPU boolean indexing)
  // Note: This creates variable-length output based on random mask
  const preIdx = preGrid.index(mask)
  const postIdx = postGrid.index(mask)

  return {
    preIndices: preIdx.astype(mx.int32),
    postIndices: postIdx.astype(mx.int32),
  }
}

/**
 * Create one-to-one connectivity - ALL ON GPU.
 */
export function createOneToOneConnectivity(
  size: number
): { preIndices: ReturnType<typeof mx.array>; postIndices: ReturnType<typeof mx.array> } {
  const indices = mx.arange(size, mx.int32)
  return { preIndices: indices, postIndices: indices }
}

/**
 * Create all-to-all connectivity - ALL ON GPU.
 */
export function createAllToAllConnectivity(
  preSize: number,
  postSize: number
): { preIndices: ReturnType<typeof mx.array>; postIndices: ReturnType<typeof mx.array> } {
  const totalSynapses = preSize * postSize

  // Pre indices: 0,0,0...1,1,1...2,2,2... (each pre connects to all post)
  const preIdx = mx.floor(mx.divide(mx.arange(totalSynapses), mx.array(postSize)))

  // Post indices: 0,1,2...0,1,2...0,1,2... (cycling through post neurons)
  const postIdx = mx.remainder(mx.arange(totalSynapses), mx.array(postSize))

  return {
    preIndices: preIdx.astype(mx.int32),
    postIndices: postIdx.astype(mx.int32),
  }
}
