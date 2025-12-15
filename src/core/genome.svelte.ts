/**
 * Genome System - Declarative Brain Specification ("DNA")
 *
 * Following sveltui pattern EXACTLY:
 * - Direct $state exports for all arrays
 * - SvelteMap for reactive registry
 * - ALL computation on GPU - NEVER convert to JS
 *
 * The genome is the "DNA" that creatures inherit:
 * - Neurons with specific types and roles
 * - Synapses with innate connectivity patterns
 * - Circuits for reflexes and drives
 * - All encoded declaratively, loaded at birth
 *
 * This enables:
 * - C. elegans: 302 named neurons with specific wiring (loaded from data)
 * - Cortical structures: populations with layer organization
 * - Any brain architecture from a single specification format
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import { SvelteMap } from 'svelte/reactivity'
import {
  allocatePopulation,
  populationSize,
  type PopulationOptions,
  type NeuronRole,
  type BrainRegion,
  NeuronTypes,
} from './neuron.svelte.ts'
import {
  allocateSynapseGroup,
  type SynapseGroupOptions,
  type SynapseType,
} from './synapse.svelte.ts'
import {
  allocateModulation,
  type ModulationOptions,
} from './modulation.svelte.ts'
import {
  allToAll,
  oneToOne,
  randomSparse,
  topographic,
  lateralInhibition,
  recurrent,
  type Connectivity,
} from './connectivity.svelte.ts'
import {
  allocateNetwork,
  addPopulationToNetwork,
  addSynapseGroupToNetwork,
} from './network.svelte.ts'

// ============================================================================
// GENOME TYPE DEFINITIONS
// ============================================================================

/**
 * Specification for a single neuron or population.
 */
export interface NeuronSpec {
  id: string
  size: number  // 1 for individual neuron, N for population
  type?: keyof typeof NeuronTypes
  excitatory?: boolean  // Override auto-detection from type
  role?: NeuronRole
  region?: BrainRegion
  noise?: number  // Background noise amplitude (default ~5)
  position?: [number, number, number]  // 3D position for spatial organization
}

/**
 * Specification for a synapse group.
 */
export interface SynapseSpec {
  id?: string  // Auto-generated if not provided
  pre: string  // Pre-synaptic neuron/population id
  post: string // Post-synaptic neuron/population id
  pattern: 'all-to-all' | 'one-to-one' | 'random' | 'topographic' | 'lateral-inhibition' | 'recurrent' | 'custom'
  density?: number  // For random pattern
  sigma?: number    // For topographic pattern
  radius?: number   // For lateral inhibition
  plastic?: boolean // Whether STDP is enabled (default true)
  synapseType?: SynapseType
  initialWeight?: number  // Uniform initial weight (overrides random)
  customConnectivity?: Connectivity  // For custom pattern
}

/**
 * Specification for a reflex arc (innate pathway).
 */
export interface ReflexSpec {
  name: string
  pathway: string[]  // Full pathway from stimulus to response (neuron ids)
  strength: number   // Connection strength
  plastic: boolean   // Whether reflex can be modified by learning
}

/**
 * Specification for a drive circuit.
 */
export interface DriveSpec {
  name: string
  sensor: string        // What internal state it monitors (population id)
  effector: string      // What behavior it promotes (population id)
  modulator: 'dopamine' | 'serotonin' | 'norepinephrine' | 'acetylcholine'
  threshold: number     // Activation threshold
  gain: number          // How strongly to modulate
}

/**
 * Complete genome specification.
 */
export interface Genome {
  name: string
  description?: string

  // Structural components
  neurons: NeuronSpec[]
  synapses: SynapseSpec[]

  // Innate behaviors (optional)
  reflexes?: ReflexSpec[]
  drives?: DriveSpec[]

  // Modulation settings (optional)
  modulation?: ModulationOptions

  // Metadata
  version?: string
  author?: string
}

// ============================================================================
// GENOME REGISTRY
// ============================================================================

export const genomeRegistry = $state({
  genomes: new SvelteMap<string, Genome>(),
})

// Track loaded genomes → network mappings
export const genomeToNetwork = $state(new SvelteMap<string, number>())
export const genomeToPopulations = $state(new SvelteMap<string, Map<string, number>>())
export const genomeToSynapses = $state(new SvelteMap<string, Map<string, number>>())

// ============================================================================
// GENOME VALIDATION
// ============================================================================

export interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
}

/**
 * Validate a genome specification.
 * Checks for consistency, missing references, etc.
 */
export function validateGenome(genome: Genome): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []

  // Check required fields
  if (!genome.name) {
    errors.push('Genome must have a name')
  }

  if (!genome.neurons || genome.neurons.length === 0) {
    errors.push('Genome must have at least one neuron')
  }

  // Build set of neuron ids and sizes
  const neuronIds = new Set<string>()
  const neuronSizes = new Map<string, number>()

  for (const neuron of genome.neurons || []) {
    if (!neuron.id) {
      errors.push('Each neuron must have an id')
    } else if (neuronIds.has(neuron.id)) {
      errors.push(`Duplicate neuron id: ${neuron.id}`)
    } else {
      neuronIds.add(neuron.id)
      neuronSizes.set(neuron.id, neuron.size)
    }

    if (neuron.size < 1) {
      errors.push(`Neuron ${neuron.id} must have size >= 1`)
    }

    if (neuron.type && !(neuron.type in NeuronTypes)) {
      errors.push(`Unknown neuron type: ${neuron.type}`)
    }
  }

  // Validate synapses reference valid neurons
  for (const synapse of genome.synapses || []) {
    if (!neuronIds.has(synapse.pre)) {
      errors.push(`Synapse references unknown pre-neuron: ${synapse.pre}`)
    }
    if (!neuronIds.has(synapse.post)) {
      errors.push(`Synapse references unknown post-neuron: ${synapse.post}`)
    }

    // Pattern-specific validation
    switch (synapse.pattern) {
      case 'random':
        if (synapse.density === undefined) {
          warnings.push(`Random synapse ${synapse.pre}→${synapse.post} without density, using 0.1`)
        } else if (synapse.density < 0 || synapse.density > 1) {
          errors.push(`Invalid density ${synapse.density} for synapse ${synapse.pre}→${synapse.post}`)
        }
        break

      case 'one-to-one':
        const preSize = neuronSizes.get(synapse.pre)
        const postSize = neuronSizes.get(synapse.post)
        if (preSize !== undefined && postSize !== undefined && preSize !== postSize) {
          errors.push(`One-to-one requires equal sizes: ${synapse.pre} (${preSize}) vs ${synapse.post} (${postSize})`)
        }
        break

      case 'topographic':
        if (synapse.sigma === undefined) {
          warnings.push(`Topographic synapse without sigma, using 2.0`)
        }
        break

      case 'lateral-inhibition':
        if (synapse.radius === undefined) {
          warnings.push(`Lateral inhibition without radius, using 1`)
        }
        if (synapse.pre !== synapse.post) {
          errors.push(`Lateral inhibition must be within same population`)
        }
        break

      case 'recurrent':
        if (synapse.pre !== synapse.post) {
          errors.push(`Recurrent pattern must be within same population`)
        }
        break

      case 'custom':
        if (!synapse.customConnectivity) {
          errors.push(`Custom pattern requires customConnectivity`)
        }
        break
    }
  }

  // Validate reflexes
  for (const reflex of genome.reflexes || []) {
    if (!reflex.pathway || reflex.pathway.length < 2) {
      errors.push(`Reflex ${reflex.name} must have pathway with at least 2 neurons`)
    }
    for (const neuronId of reflex.pathway) {
      if (!neuronIds.has(neuronId)) {
        errors.push(`Reflex ${reflex.name} references unknown neuron: ${neuronId}`)
      }
    }
    if (reflex.strength === undefined) {
      errors.push(`Reflex ${reflex.name} must have strength`)
    }
    if (reflex.plastic === undefined) {
      warnings.push(`Reflex ${reflex.name} missing plastic flag, defaulting to false`)
    }
  }

  // Validate drives
  for (const drive of genome.drives || []) {
    if (!neuronIds.has(drive.sensor)) {
      errors.push(`Drive ${drive.name} references unknown sensor: ${drive.sensor}`)
    }
    if (!neuronIds.has(drive.effector)) {
      errors.push(`Drive ${drive.name} references unknown effector: ${drive.effector}`)
    }
    if (drive.threshold === undefined) {
      errors.push(`Drive ${drive.name} must have threshold`)
    }
    if (drive.gain === undefined) {
      errors.push(`Drive ${drive.name} must have gain`)
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  }
}

// ============================================================================
// GENOME LOADING
// ============================================================================

/**
 * Load a genome and create the corresponding network.
 * Returns the network index.
 *
 * This is the main entry point for creating a brain from DNA.
 */
export function loadGenome(genome: Genome): number {
  // Validate first
  const validation = validateGenome(genome)
  if (!validation.valid) {
    throw new Error(`Invalid genome:\n  ${validation.errors.join('\n  ')}`)
  }

  // Log warnings
  for (const warning of validation.warnings) {
    console.warn(`Genome warning: ${warning}`)
  }

  // Store genome
  genomeRegistry.genomes.set(genome.name, genome)

  // Create network
  const networkIndex = allocateNetwork(genome.name)

  // Create modulation system
  allocateModulation(genome.name, genome.modulation)

  // Track mappings
  const popMap = new Map<string, number>()
  const synMap = new Map<string, number>()
  genomeToNetwork.set(genome.name, networkIndex)
  genomeToPopulations.set(genome.name, popMap)
  genomeToSynapses.set(genome.name, synMap)

  // Build neuron size map for synapse creation
  const neuronSizes = new Map<string, number>()
  for (const spec of genome.neurons) {
    neuronSizes.set(spec.id, spec.size)
  }

  // Create neurons/populations
  for (const neuronSpec of genome.neurons) {
    const options: PopulationOptions = {
      type: neuronSpec.type,
      excitatory: neuronSpec.excitatory,
      role: neuronSpec.role,
      region: neuronSpec.region,
      noise: neuronSpec.noise,
    }

    const popIndex = allocatePopulation(
      `${genome.name}_${neuronSpec.id}`,
      neuronSpec.size,
      options
    )

    popMap.set(neuronSpec.id, popIndex)
    addPopulationToNetwork(networkIndex, popIndex)
  }

  // Create synapses
  for (let i = 0; i < genome.synapses.length; i++) {
    const spec = genome.synapses[i]
    const synapseId = spec.id ?? `syn_${spec.pre}_${spec.post}_${i}`

    const prePopIndex = popMap.get(spec.pre)!
    const postPopIndex = popMap.get(spec.post)!
    const preSize = neuronSizes.get(spec.pre)!
    const postSize = neuronSizes.get(spec.post)!

    // Create connectivity based on pattern
    let connectivity: Connectivity

    switch (spec.pattern) {
      case 'all-to-all':
        connectivity = allToAll(preSize, postSize)
        break

      case 'one-to-one':
        connectivity = oneToOne(preSize)
        break

      case 'random':
        connectivity = randomSparse(preSize, postSize, spec.density ?? 0.1)
        break

      case 'topographic':
        connectivity = topographic(
          preSize, 1,  // Assume 1D for now
          postSize, 1,
          spec.sigma ?? 2.0,
          spec.density ?? 0.5
        )
        break

      case 'lateral-inhibition':
        connectivity = lateralInhibition(preSize, spec.radius ?? 1, false)
        break

      case 'recurrent':
        connectivity = recurrent(preSize, spec.density ?? 0.1, false)
        break

      case 'custom':
        connectivity = spec.customConnectivity!
        break

      default:
        throw new Error(`Unknown synapse pattern: ${spec.pattern}`)
    }

    // Create synapse group options
    const options: SynapseGroupOptions = {
      plastic: spec.plastic ?? true,
      synapseType: spec.synapseType,
    }

    // Set initial weights if specified
    if (spec.initialWeight !== undefined) {
      mx.eval(connectivity.preIndices)
      const numSynapses = connectivity.preIndices.shape[0]
      options.initialWeights = mx.full([numSynapses], spec.initialWeight, mx.float32)
    }

    const groupIndex = allocateSynapseGroup(
      `${genome.name}_${synapseId}`,
      prePopIndex,
      postPopIndex,
      connectivity.preIndices,
      connectivity.postIndices,
      options
    )

    synMap.set(synapseId, groupIndex)
    addSynapseGroupToNetwork(networkIndex, groupIndex)
  }

  // Create reflex pathways
  if (genome.reflexes) {
    for (const reflex of genome.reflexes) {
      createReflexFromSpec(genome.name, reflex, popMap, synMap, neuronSizes, networkIndex)
    }
  }

  return networkIndex
}

/**
 * Create a reflex arc from specification.
 * Creates synapses between consecutive neurons in the pathway.
 */
function createReflexFromSpec(
  genomeName: string,
  reflex: ReflexSpec,
  popMap: Map<string, number>,
  synMap: Map<string, number>,
  neuronSizes: Map<string, number>,
  networkIndex: number
) {
  for (let i = 0; i < reflex.pathway.length - 1; i++) {
    const preId = reflex.pathway[i]
    const postId = reflex.pathway[i + 1]
    const synapseKey = `reflex_${reflex.name}_${preId}_${postId}`

    // Skip if this connection already exists
    if (synMap.has(synapseKey)) continue

    const prePopIndex = popMap.get(preId)!
    const postPopIndex = popMap.get(postId)!
    const preSize = neuronSizes.get(preId)!
    const postSize = neuronSizes.get(postId)!

    // Create connectivity - all-to-all for reflex reliability
    const connectivity = allToAll(preSize, postSize)
    mx.eval(connectivity.preIndices)
    const numSynapses = connectivity.preIndices.shape[0]

    const groupIndex = allocateSynapseGroup(
      `${genomeName}_${synapseKey}`,
      prePopIndex,
      postPopIndex,
      connectivity.preIndices,
      connectivity.postIndices,
      {
        plastic: reflex.plastic,
        initialWeights: mx.full([numSynapses], reflex.strength, mx.float32),
      }
    )

    synMap.set(synapseKey, groupIndex)
    addSynapseGroupToNetwork(networkIndex, groupIndex)
  }
}

// ============================================================================
// GENOME ACCESS
// ============================================================================

/**
 * Get the network index for a loaded genome.
 */
export function getNetworkForGenome(genomeName: string): number | undefined {
  return genomeToNetwork.get(genomeName)
}

/**
 * Get the population index for a neuron in a loaded genome.
 */
export function getPopulationForNeuron(genomeName: string, neuronId: string): number | undefined {
  return genomeToPopulations.get(genomeName)?.get(neuronId)
}

/**
 * Get all population indices for a loaded genome.
 */
export function getPopulationsForGenome(genomeName: string): Map<string, number> | undefined {
  return genomeToPopulations.get(genomeName)
}

/**
 * Get the synapse group index for a synapse in a loaded genome.
 */
export function getSynapseGroupForSynapse(genomeName: string, synapseId: string): number | undefined {
  return genomeToSynapses.get(genomeName)?.get(synapseId)
}

/**
 * Get all synapse group indices for a loaded genome.
 */
export function getSynapseGroupsForGenome(genomeName: string): Map<string, number> | undefined {
  return genomeToSynapses.get(genomeName)
}

/**
 * Get the loaded genome specification.
 */
export function getGenome(genomeName: string): Genome | undefined {
  return genomeRegistry.genomes.get(genomeName)
}

/**
 * Check if a genome is loaded.
 */
export function isGenomeLoaded(genomeName: string): boolean {
  return genomeToNetwork.has(genomeName)
}
