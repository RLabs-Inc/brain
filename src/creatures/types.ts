/**
 * Creature Types - Brain + Body Interface
 *
 * A creature is a brain (network of neurons) connected to a body (sensors + motors).
 *
 * CRITICAL INSIGHT: Nothing is born a blank slate.
 * DNA encodes innate wiring - reflexes, drives, biases.
 * STDP + reward learning REFINES what DNA provided.
 * The factory function IS the creature's "DNA" - specific wiring, not random.
 *
 * NOTE: This file contains TypeScript interfaces (compile-time contracts).
 * Actual reactive state lives in the creature implementations (.svelte.ts files).
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

// ============================================================================
// CREATURE STATE (for visualization and data collection)
// ============================================================================

export interface CreatureState {
  // Identity
  id: string

  // Neural activity summary (GPU → JS for display only)
  totalSpikes: number
  activePopulations: number

  // Per-population activity (for detailed viz)
  populationActivity: Map<string, number>  // popName → spike count

  // Sensor states
  sensorValues: Map<string, number>  // sensorName → current value

  // Motor outputs
  motorOutputs: Map<string, number>  // motorName → current output

  // Learning state
  currentReward: number
  totalRewardReceived: number
}

// ============================================================================
// CREATURE INTERFACE
// ============================================================================

export interface Creature {
  // Identity
  readonly id: string

  // Network structure (indices into core module arrays)
  readonly networkIndex: number
  readonly populationIndices: number[]
  readonly synapseGroupIndices: number[]
  readonly sensorIndices: number[]
  readonly motorIndices: number[]

  // Named access (for semantic operations)
  readonly populations: Map<string, number>   // name → popIndex
  readonly sensors: Map<string, number>       // name → sensorIndex
  readonly motors: Map<string, number>        // name → motorIndex

  /**
   * Inject sensory input into the creature's brain.
   * World → Brain pathway.
   *
   * @param stimuli - Map of sensor name → stimulus value
   */
  sense(stimuli: Map<string, number>): void

  /**
   * Run one timestep of neural computation.
   * This is where thinking happens - transmission, STDP, integration.
   *
   * @param dt - Timestep in ms (typically 1.0)
   */
  think(dt: number): Promise<void>

  /**
   * Read motor outputs from the creature's brain.
   * Brain → World pathway.
   *
   * @returns Map of motor name → action value
   */
  act(): Map<string, number>

  /**
   * Set the reward/punishment signal.
   * Modulates STDP learning via dopamine.
   *
   * @param value - Reward (positive) or punishment (negative)
   */
  setReward(value: number): void

  /**
   * Get current state for visualization/data collection.
   * This is the ONLY place we do GPU → JS conversion.
   */
  getState(): Promise<CreatureState>

  /**
   * Release all allocated resources.
   * Must be called when creature is no longer needed.
   */
  destroy(): void
}

// ============================================================================
// CREATURE FACTORY
// ============================================================================

export interface CreatureOptions {
  // Seed for reproducible "genetics" (if creature has any randomness)
  seed?: number

  // Learning parameters (can override defaults)
  learningEnabled?: boolean
  stdpRate?: number
  rewardModulation?: number

  // Scale factor (for testing at different sizes)
  // 1.0 = C. elegans scale (~300 neurons)
  // 0.1 = mini version for quick tests (~30 neurons)
  scale?: number
}

/**
 * Factory function signature for creatures.
 * The factory IS the creature's "DNA" - it encodes the innate wiring.
 */
export type CreatureFactory = (id: string, options?: CreatureOptions) => Creature

// ============================================================================
// INNATE WIRING TYPES (the "DNA")
// ============================================================================

/**
 * Describes a population in the creature's brain.
 * Part of the innate specification.
 */
export interface PopulationSpec {
  name: string
  size: number
  type: 'RS' | 'IB' | 'CH' | 'FS' | 'LTS' | 'TC' | 'RZ'  // Izhikevich types
  role: 'sensory' | 'inter' | 'motor' | 'modulatory'
}

/**
 * Describes innate connectivity between populations.
 * This is what DNA encodes - NOT random.
 */
export interface ConnectionSpec {
  from: string           // Population name
  to: string             // Population name
  pattern: 'all_to_all' | 'one_to_one' | 'random' | 'topographic'
  probability?: number   // For random pattern
  weight: number         // Initial weight (innate strength)
  plastic: boolean       // Can this connection learn via STDP?
  delay?: number         // Synaptic delay in ms
}

/**
 * Describes a sensor attached to a population.
 */
export interface SensorSpec {
  name: string
  population: string     // Which population receives this input
  type: 'touch' | 'chemical' | 'light' | 'temperature' | 'proprioception' | 'pain'
  encoding: 'rate' | 'population' | 'temporal' | 'place'
  gain?: number
}

/**
 * Describes a motor output from a population.
 */
export interface MotorSpec {
  name: string
  population: string     // Which population drives this output
  type: 'muscle' | 'gland' | 'behavior' | 'locomotion'
  decoding: 'rate' | 'population' | 'winner_take_all' | 'labeled_line'
  gain?: number
}

/**
 * Complete innate specification for a creature.
 * This IS the creature's genome.
 */
export interface CreatureGenome {
  name: string
  description: string

  // Brain structure
  populations: PopulationSpec[]
  connections: ConnectionSpec[]

  // Body interface
  sensors: SensorSpec[]
  motors: MotorSpec[]
}
