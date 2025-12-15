/**
 * Network - Orchestration Layer
 *
 * Following sveltui pattern EXACTLY:
 * - Direct $state exports for all arrays
 * - SvelteMap for reactive registry
 * - Free index pool for reuse
 * - Double function pattern ONLY for derived values
 * - ALL computation on GPU - NEVER convert to JS
 * - Direct mutation of state arrays
 *
 * The network provides:
 * 1. Grouping of populations and synapse groups
 * 2. Integration with neuromodulation system
 * 3. E/I balance monitoring and homeostatic plasticity
 * 4. Step function for external drivers to advance simulation
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import { SvelteMap } from 'svelte/reactivity'
import {
  registry as neuronRegistry,
  populationSize,
  voltage,
  recovery,
  current,
  fired,
  integrate,
  resetPopulation,
  getNeuronDerived,
  allocatePopulation,
  isExcitatory,
  THRESHOLD,
} from './neuron.svelte.ts'

// ============================================================================
// CACHED CONSTANTS (avoid creating new Metal buffers every call!)
// ============================================================================

const CONST_ZERO_F32 = mx.array(0, mx.float32)
const CONST_ZERO_I32 = mx.array(0, mx.int32)
const CONST_ONE_F32 = mx.array(1, mx.float32)
const CONST_ONE_I32 = mx.array(1, mx.int32)
const CONST_EPSILON = mx.array(1e-8, mx.float32)
const CONST_REWARD_THRESHOLD = mx.array(0.001, mx.float32)
const CONST_SCALE_MIN = mx.array(0.5, mx.float32)
const CONST_SCALE_MAX = mx.array(2.0, mx.float32)
import {
  registry as synapseRegistry,
  groupPrePopIndex,
  groupPostPopIndex,
  weights as groupWeights,
  minWeight as groupMinWeight,
  maxWeight as groupMaxWeight,
  transmit,
  updateTraces,
  applySTDP,
  applyReward,
  resetLearning,
  allocateSynapseGroup,
  createRandomConnectivity,
} from './synapse.svelte.ts'
import {
  decayModulatorsById,
  getPlasticityGateById,
  isModulationAllocated,
} from './modulation.svelte.ts'

// ============================================================================
// NETWORK REGISTRY (like sveltui engine.svelte.ts registry)
// ============================================================================

export const registry = $state({
  idToIndex: new SvelteMap<string, number>(),
  indexToId: new SvelteMap<number, string>(),
  allocatedIndices: new Set<number>(),
  freeIndices: [] as number[],
  nextIndex: 0,
})

// ============================================================================
// NETWORK STATE (DIRECT EXPORTS)
// ============================================================================

// Network membership - which populations and synapse groups belong to each network
export const networkPopulations = $state<number[][]>([])     // [networkIndex] → [popIndex, ...]
export const networkSynapseGroups = $state<number[][]>([])   // [networkIndex] → [groupIndex, ...]

// Global dopamine/reward signal per network - GPU scalars!
// NOTE: This is kept for backwards compatibility. Prefer using modulation.svelte.ts
export const dopamine = $state<ReturnType<typeof mx.array>[]>([])

// Dopamine decay factor per network - GPU scalar
export const dopamineDecay = $state<ReturnType<typeof mx.array>[]>([])

// Timestep counter (GPU scalar for consistency)
export const timestep = $state<ReturnType<typeof mx.array>[]>([])

// ============================================================================
// HOMEOSTATIC PLASTICITY STATE
// ============================================================================

// Target firing rate for homeostasis (spikes per timestep per neuron)
export const targetFiringRate = $state<ReturnType<typeof mx.array>[]>([])

// Homeostatic scaling factor per population (GPU arrays)
// Multiplied by synaptic weights to maintain target activity
export const homeostaticScale = $state<ReturnType<typeof mx.array>[][]>([])

// Time constant for homeostatic plasticity (how fast to adjust)
export const homeostaticTau = $state<ReturnType<typeof mx.array>[]>([])

// Running average firing rate per population (GPU arrays)
export const avgFiringRate = $state<ReturnType<typeof mx.array>[][]>([])

// Whether homeostasis is enabled per network
export const homeostaticEnabled = $state<boolean[]>([])

// E/I balance tracking
export const excitatorySpikeCount = $state<ReturnType<typeof mx.array>[]>([])
export const inhibitorySpikeCount = $state<ReturnType<typeof mx.array>[]>([])

// ============================================================================
// NETWORK MANAGEMENT (like sveltui allocateIndex/releaseIndex)
// ============================================================================

/**
 * Allocate a new network.
 */
export function allocateNetwork(id: string): number {
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

  // Initialize network state
  networkPopulations[index] = []
  networkSynapseGroups[index] = []
  dopamine[index] = mx.array(0, mx.float32)
  dopamineDecay[index] = mx.array(0.9, mx.float32)
  timestep[index] = mx.array(0, mx.int32)

  // Initialize homeostatic state
  targetFiringRate[index] = mx.array(0.05, mx.float32)  // 5% firing rate target
  homeostaticScale[index] = []  // Will be populated when populations are added
  homeostaticTau[index] = mx.array(1000, mx.float32)    // Slow adaptation (1000 timesteps)
  avgFiringRate[index] = []     // Will be populated when populations are added
  homeostaticEnabled[index] = false  // Disabled by default

  // Initialize E/I balance tracking
  excitatorySpikeCount[index] = mx.array(0, mx.float32)
  inhibitorySpikeCount[index] = mx.array(0, mx.float32)

  return index
}

/**
 * Get network index by id.
 */
export function getNetworkIndex(id: string): number | undefined {
  return registry.idToIndex.get(id)
}

/**
 * Add a population to a network.
 */
export function addPopulationToNetwork(networkIndex: number, popIndex: number) {
  if (!networkPopulations[networkIndex].includes(popIndex)) {
    networkPopulations[networkIndex].push(popIndex)

    // Initialize homeostatic state for this population
    const size = populationSize[popIndex]
    const localIndex = networkPopulations[networkIndex].length - 1
    homeostaticScale[networkIndex][localIndex] = mx.ones([size], mx.float32)
    avgFiringRate[networkIndex][localIndex] = mx.full([size], 0.05, mx.float32)  // Initialize to target
  }
}

/**
 * Add a synapse group to a network.
 */
export function addSynapseGroupToNetwork(networkIndex: number, groupIndex: number) {
  if (!networkSynapseGroups[networkIndex].includes(groupIndex)) {
    networkSynapseGroups[networkIndex].push(groupIndex)
  }
}

/**
 * Release a network.
 */
export function releaseNetwork(id: string): void {
  const index = registry.idToIndex.get(id)
  if (index === undefined) return

  // Update registry
  registry.idToIndex.delete(id)
  registry.indexToId.delete(index)
  registry.allocatedIndices.delete(index)
  registry.freeIndices.push(index)

  // Clean up
  networkPopulations[index] = []
  networkSynapseGroups[index] = []
  dopamine[index] = mx.array(0, mx.float32)
  dopamineDecay[index] = mx.array(0.9, mx.float32)
  timestep[index] = mx.array(0, mx.int32)

  // Clean up homeostatic state
  targetFiringRate[index] = mx.array(0.05, mx.float32)
  homeostaticScale[index] = []
  homeostaticTau[index] = mx.array(1000, mx.float32)
  avgFiringRate[index] = []
  homeostaticEnabled[index] = false

  // Clean up E/I tracking
  excitatorySpikeCount[index] = mx.array(0, mx.float32)
  inhibitorySpikeCount[index] = mx.array(0, mx.float32)
}

// ============================================================================
// NETWORK OPERATIONS
// These are called by external drivers - NOT controlled internally
// NOTE: We iterate over populations/groups in JS because they are structural
// (small number of items). The COMPUTATION within each is all GPU.
// ============================================================================

/**
 * Process one simulation step for a network.
 *
 * Order of operations:
 * 1. Transmit spikes (from neurons that fired LAST step)
 * 2. Update STDP traces
 * 3. Apply STDP (update eligibility)
 * 4. Apply reward signal (from modulation or legacy dopamine)
 * 5. Integrate neurons (advance dynamics)
 * 6. Track E/I balance
 * 7. Update homeostatic plasticity (every N steps)
 * 8. Decay modulators
 * 9. Increment timestep counter
 *
 * This does NOT control time - it's called BY an external driver.
 * The JS loops here iterate over STRUCTURE (small number of pops/groups).
 * ALL actual computation happens on GPU inside each function.
 */
export function step(networkIndex: number, dt: number = 1.0) {
  const pops = networkPopulations[networkIndex]
  const groups = networkSynapseGroups[networkIndex]
  const networkId = registry.indexToId.get(networkIndex)

  // 1. Transmit spikes from all synapse groups (GPU computation inside)
  for (const groupIndex of groups) {
    transmit(groupIndex)
  }

  // 2. Update STDP traces (GPU computation inside)
  for (const groupIndex of groups) {
    updateTraces(groupIndex)
  }

  // 3. Apply STDP - updates eligibility, not weights directly (GPU inside)
  for (const groupIndex of groups) {
    applySTDP(groupIndex, false)
  }

  // 4. Apply reward signal
  // Use modulation system if available, otherwise fall back to legacy dopamine
  let rewardSignal: ReturnType<typeof mx.array>
  const useModulation = networkId && isModulationAllocated(networkId)

  if (useModulation) {
    // Use combined plasticity gate from modulation system
    rewardSignal = getPlasticityGateById(networkId!)
  } else {
    // Legacy: use network-level dopamine
    rewardSignal = dopamine[networkIndex]
  }

  // Check if there's any reward signal - wrap in tidy to clean up intermediate
  const shouldApplyReward = mx.tidy(() => {
    const hasReward = mx.greater(mx.abs(rewardSignal), CONST_REWARD_THRESHOLD)
    mx.eval(hasReward)
    return hasReward
  })

  if (shouldApplyReward.item()) {
    for (const groupIndex of groups) {
      applyReward(groupIndex, rewardSignal)
    }
    // Decay legacy dopamine if used
    if (!useModulation) {
      dopamine[networkIndex] = mx.multiply(dopamine[networkIndex], dopamineDecay[networkIndex])
    }
  }

  // 5. Integrate all populations (GPU computation inside)
  for (const popIndex of pops) {
    integrate(popIndex, dt)
  }

  // 6. Track E/I balance
  updateEIBalance(networkIndex)

  // 7. Apply homeostatic plasticity periodically (every 100 steps)
  mx.eval(timestep[networkIndex])
  const t = timestep[networkIndex].item() as number
  if (homeostaticEnabled[networkIndex] && t > 0 && t % 100 === 0) {
    applyHomeostasis(networkIndex)
  }

  // 8. Decay modulators if using modulation system
  if (useModulation) {
    decayModulatorsById(networkId!)
  }

  // 9. Increment timestep (GPU add) - use cached constant!
  timestep[networkIndex] = mx.add(timestep[networkIndex], CONST_ONE_I32)

  // 10. CRITICAL: Force evaluation to release intermediate tensors
  // Without this, MLX accumulates lazy operations and runs out of tracked resources
  // This synchronizes GPU computation and allows memory to be freed
  const evalTargets: any[] = []
  for (const popIndex of pops) {
    evalTargets.push(voltage[popIndex], recovery[popIndex], current[popIndex], fired[popIndex])
  }
  if (evalTargets.length > 0) {
    mx.eval(...evalTargets)
  }
}

/**
 * Set the dopamine/reward signal for a network.
 * This modulates learning via eligibility traces.
 * Takes a GPU scalar for consistency.
 */
export function setReward(networkIndex: number, reward: ReturnType<typeof mx.array>) {
  dopamine[networkIndex] = reward
}

/**
 * Set reward from a number (convenience function).
 */
export function setRewardValue(networkIndex: number, reward: number) {
  dopamine[networkIndex] = mx.array(reward, mx.float32)
}

/**
 * Set dopamine decay rate.
 */
export function setDopamineDecay(networkIndex: number, decay: number) {
  dopamineDecay[networkIndex] = mx.array(decay, mx.float32)
}

/**
 * Reset entire network to initial state.
 */
export function resetNetwork(networkIndex: number) {
  const pops = networkPopulations[networkIndex]
  const groups = networkSynapseGroups[networkIndex]

  // Reset all populations
  for (const popIndex of pops) {
    resetPopulation(popIndex)
  }

  // Reset learning state (not weights)
  for (const groupIndex of groups) {
    resetLearning(groupIndex)
  }

  // Reset network state
  dopamine[networkIndex] = mx.array(0, mx.float32)
  timestep[networkIndex] = mx.array(0, mx.int32)
}

/**
 * Force GPU evaluation for all network state.
 * Useful for synchronization points.
 */
export async function evaluate(networkIndex: number) {
  const pops = networkPopulations[networkIndex]

  // Collect all GPU arrays that need evaluation
  const arrays: ReturnType<typeof mx.array>[] = []
  for (const popIndex of pops) {
    arrays.push(voltage[popIndex])
    arrays.push(current[popIndex])
  }

  if (arrays.length > 0) {
    await mx.asyncEval(...arrays)
  }
}

// ============================================================================
// E/I BALANCE TRACKING
// ============================================================================

/**
 * Update E/I balance tracking.
 * Counts spikes from excitatory vs inhibitory populations.
 * Called automatically by step().
 */
function updateEIBalance(networkIndex: number) {
  const pops = networkPopulations[networkIndex]

  // Use mx.tidy to clean up intermediate tensors from the loop
  const result = mx.tidy(() => {
    let excSpikes = CONST_ZERO_F32
    let inhSpikes = CONST_ZERO_F32

    for (const popIndex of pops) {
      const spikeCount = mx.sum(fired[popIndex])
      if (isExcitatory[popIndex]) {
        excSpikes = mx.add(excSpikes, spikeCount)
      } else {
        inhSpikes = mx.add(inhSpikes, spikeCount)
      }
    }

    mx.eval(excSpikes, inhSpikes)
    return { excSpikes, inhSpikes }
  })

  excitatorySpikeCount[networkIndex] = result.excSpikes
  inhibitorySpikeCount[networkIndex] = result.inhSpikes
}

/**
 * Get the current E/I ratio for a network.
 * Healthy networks typically have ~80/20 E/I ratio.
 * Returns GPU scalar (excitatory / (excitatory + inhibitory))
 */
export function getEIRatio(networkIndex: number): ReturnType<typeof mx.array> {
  const exc = excitatorySpikeCount[networkIndex]
  const inh = inhibitorySpikeCount[networkIndex]
  const total = mx.add(exc, inh)

  // Avoid division by zero - use cached constant!
  return mx.divide(exc, mx.add(total, CONST_EPSILON))
}

/**
 * Get E/I balance stats.
 * Returns object with excitatory count, inhibitory count, and ratio.
 */
export function getEIBalanceStats(networkIndex: number) {
  return {
    excitatory: excitatorySpikeCount[networkIndex],
    inhibitory: inhibitorySpikeCount[networkIndex],
    ratio: getEIRatio(networkIndex),
  }
}

// ============================================================================
// HOMEOSTATIC PLASTICITY
// ============================================================================

/**
 * Enable homeostatic plasticity for a network.
 *
 * @param networkIndex - Network index
 * @param targetRate - Target firing rate (0-1, default 0.05 = 5%)
 * @param tau - Time constant in timesteps (default 1000)
 */
export function enableHomeostasis(
  networkIndex: number,
  targetRate: number = 0.05,
  tau: number = 1000
) {
  homeostaticEnabled[networkIndex] = true
  targetFiringRate[networkIndex] = mx.array(targetRate, mx.float32)
  homeostaticTau[networkIndex] = mx.array(tau, mx.float32)
}

/**
 * Disable homeostatic plasticity for a network.
 */
export function disableHomeostasis(networkIndex: number) {
  homeostaticEnabled[networkIndex] = false
}

/**
 * Apply homeostatic plasticity to maintain target firing rate.
 * Scales synaptic weights to bring activity back to target.
 * Called periodically by step() when enabled.
 *
 * Uses multiplicative scaling:
 * - If firing too high: scale down weights
 * - If firing too low: scale up weights
 */
function applyHomeostasis(networkIndex: number) {
  const pops = networkPopulations[networkIndex]
  const groups = networkSynapseGroups[networkIndex]

  const target = targetFiringRate[networkIndex]
  const tau = homeostaticTau[networkIndex]

  // Use mx.tidy to clean up all intermediate tensors
  const newAvgRates: ReturnType<typeof mx.array>[] = []
  const newScales: ReturnType<typeof mx.array>[] = []

  // Update running average firing rate for each population
  for (let i = 0; i < pops.length; i++) {
    const popIndex = pops[i]
    const size = populationSize[popIndex]

    const result = mx.tidy(() => {
      // Learning rate
      const learningRate = mx.divide(CONST_ONE_F32, tau)

      // Current instantaneous firing rate
      const firedFloat = mx.where(
        fired[popIndex],
        mx.ones([size], mx.float32),
        mx.zeros([size], mx.float32)
      )

      // Exponential moving average - use cached constants!
      const alpha = learningRate
      const oneMinusAlpha = mx.subtract(CONST_ONE_F32, alpha)
      const newAvg = mx.add(
        mx.multiply(oneMinusAlpha, avgFiringRate[networkIndex][i]),
        mx.multiply(alpha, firedFloat)
      )

      // Compute scaling factor: target / actual - use cached epsilon!
      const scale = mx.divide(target, mx.add(mx.mean(newAvg), CONST_EPSILON))

      // Clamp scale to prevent runaway - use cached constants!
      const clampedScale = mx.clip(scale, CONST_SCALE_MIN, CONST_SCALE_MAX)

      const newScale = mx.multiply(
        homeostaticScale[networkIndex][i],
        mx.power(clampedScale, learningRate)  // Gradual adjustment
      )

      mx.eval(newAvg, newScale)
      return { newAvg, newScale }
    })

    newAvgRates.push(result.newAvg)
    newScales.push(result.newScale)
  }

  // Update state arrays
  for (let i = 0; i < pops.length; i++) {
    avgFiringRate[networkIndex][i] = newAvgRates[i]
    homeostaticScale[networkIndex][i] = newScales[i]
  }

  // Apply scaling to incoming synapses of each population
  for (const groupIndex of groups) {
    const postPopIndex = groupPostPopIndex[groupIndex]

    // Find which local index this population has
    const localIndex = pops.indexOf(postPopIndex)
    if (localIndex < 0) continue

    // Use mx.tidy for the weight scaling
    const newWeights = mx.tidy(() => {
      // Scale weights by homeostatic factor
      const scale = mx.mean(homeostaticScale[networkIndex][localIndex])

      // Multiply weights by scale, keeping them within bounds
      const scaledWeights = mx.multiply(groupWeights[groupIndex], scale)
      const clipped = mx.clip(
        scaledWeights,
        groupMinWeight[groupIndex],
        groupMaxWeight[groupIndex]
      )
      mx.eval(clipped)
      return clipped
    })

    groupWeights[groupIndex] = newWeights
  }
}

/**
 * Reset homeostatic state for a network.
 * Call this when starting a new training episode.
 */
export function resetHomeostasis(networkIndex: number) {
  const pops = networkPopulations[networkIndex]
  const target = targetFiringRate[networkIndex].item() as number

  for (let i = 0; i < pops.length; i++) {
    const size = populationSize[pops[i]]
    homeostaticScale[networkIndex][i] = mx.ones([size], mx.float32)
    avgFiringRate[networkIndex][i] = mx.full([size], target, mx.float32)
  }
}

/**
 * Get homeostasis stats for debugging.
 */
export function getHomeostasisStats(networkIndex: number) {
  const pops = networkPopulations[networkIndex]

  const scales: ReturnType<typeof mx.array>[] = []
  const rates: ReturnType<typeof mx.array>[] = []

  for (let i = 0; i < pops.length; i++) {
    scales.push(mx.mean(homeostaticScale[networkIndex][i]))
    rates.push(mx.mean(avgFiringRate[networkIndex][i]))
  }

  return {
    enabled: homeostaticEnabled[networkIndex],
    targetRate: targetFiringRate[networkIndex],
    avgScales: scales,
    avgRates: rates,
  }
}

// ============================================================================
// DERIVED VALUES - Double function pattern
// ALL values stay on GPU!
// ============================================================================

/**
 * Get derived values for a network.
 * ALL values are GPU arrays/scalars - no .item() conversions!
 */
export function getNetworkDerived(networkIndex: number) {
  const pops = networkPopulations[networkIndex]

  // Total neuron count in network (can compute on CPU - it's metadata)
  const totalNeurons = $derived(
    pops.reduce((sum, popIndex) => sum + populationSize[popIndex], 0)
  )

  // Total spike count across all populations - STAYS ON GPU!
  // We sum fired arrays and then sum across populations
  const totalSpikes = $derived.by(() => {
    if (pops.length === 0) return mx.array(0, mx.int32)

    // Sum fired neurons in each population, then sum those sums
    let total = mx.array(0, mx.int32)
    for (const popIndex of pops) {
      const popSpikes = mx.sum(fired[popIndex])
      total = mx.add(total, popSpikes)
    }
    return total
  })

  // Average voltage across all populations - STAYS ON GPU!
  const avgVoltage = $derived.by(() => {
    if (pops.length === 0) return mx.array(-70, mx.float32)

    // Concatenate all voltages and take mean
    const allVoltages = pops.map(popIndex => voltage[popIndex])
    const concatenated = mx.concatenate(allVoltages)
    return mx.mean(concatenated)
  })

  // E/I ratio
  const eiRatio = $derived(getEIRatio(networkIndex))

  return () => ({
    totalNeurons,   // JS number (metadata)
    totalSpikes,    // GPU scalar
    avgVoltage,     // GPU scalar
    timestep: timestep[networkIndex],   // GPU scalar
    dopamine: dopamine[networkIndex],   // GPU scalar
    eiRatio,        // GPU scalar (0-1, healthy ~0.8)
    excSpikes: excitatorySpikeCount[networkIndex],  // GPU scalar
    inhSpikes: inhibitorySpikeCount[networkIndex],  // GPU scalar
  })
}

// ============================================================================
// UTILITY: Quick network setup
// ============================================================================

/**
 * Create a simple feedforward network.
 * Returns the network index and population indices.
 *
 * This is a convenience function - you can also build networks manually.
 * ALL connectivity creation is on GPU.
 */
export function createFeedforwardNetwork(
  id: string,
  layerSizes: number[],
  connectionDensity: number = 0.1
): {
  networkIndex: number
  populationIndices: number[]
} {
  const networkIndex = allocateNetwork(id)
  const populationIndices: number[] = []

  // Create populations for each layer
  for (let i = 0; i < layerSizes.length; i++) {
    const popIndex = allocatePopulation(`${id}_layer${i}`, layerSizes[i])
    addPopulationToNetwork(networkIndex, popIndex)
    populationIndices.push(popIndex)
  }

  // Connect consecutive layers (GPU connectivity creation)
  for (let i = 0; i < layerSizes.length - 1; i++) {
    const preIndex = populationIndices[i]
    const postIndex = populationIndices[i + 1]

    // Create connectivity on GPU
    const conn = createRandomConnectivity(layerSizes[i], layerSizes[i + 1], connectionDensity)

    const groupIndex = allocateSynapseGroup(
      `${id}_syn${i}_${i + 1}`,
      preIndex,
      postIndex,
      conn.preIndices,   // Already GPU array
      conn.postIndices   // Already GPU array
    )
    addSynapseGroupToNetwork(networkIndex, groupIndex)
  }

  return { networkIndex, populationIndices }
}
