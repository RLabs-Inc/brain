/**
 * Worm Creature - C. elegans Inspired Navigation Circuit
 *
 * Based on Gray, Hill & Bargmann (2005) "A circuit for navigation in C. elegans"
 * PNAS 102(9):3184-3191
 *
 * KEY BIOLOGICAL INSIGHT: Nothing is born a blank slate.
 * This wiring is what evolution discovered over millions of years.
 * STDP can refine it, but the innate circuit IS the starting point.
 *
 * Circuit architecture (simplified from actual C. elegans):
 *
 * SENSORY LAYER:
 *   - Chemical sensors (AWC/ASK-like): detect food gradient
 *   - Left/Right bilateral sensors for gradient direction
 *
 * INTERNEURON LAYER (decision):
 *   - AIB-like: promotes turns/reversals (local search)
 *   - AIY-like: suppresses turns (dispersal/forward movement)
 *   - The BALANCE between these determines behavior
 *
 * MOTOR LAYER:
 *   - Forward neurons: drive forward locomotion
 *   - Turn neurons: drive turning (omega bends)
 *   - (Simplified from actual AVA/AVB/SMD/RIV)
 *
 * INNATE WIRING (the "DNA"):
 *   - Chemical → AIB (excitatory): food detection promotes turning
 *   - Chemical → AIY (inhibitory): food detection suppresses forward run
 *   - AIB → Turn (excitatory): AIB activity causes turns
 *   - AIY → Forward (excitatory): AIY activity causes forward movement
 *   - AIB ←→ AIY (mutual inhibition): winner-take-all competition
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import type { Creature, CreatureState, CreatureOptions } from './types.ts'
import {
  allocatePopulation,
  releasePopulation,
  integrate,
  injectCurrent,
  voltage,
  fired,
  populationSize,
  registry as neuronRegistry,
} from '../core/neuron.svelte.ts'
import {
  allocateSynapseGroup,
  releaseSynapseGroup,
  transmit,
  updateTraces,
  applySTDP,
  applyReward,
  createAllToAllConnectivity,
  weights,
} from '../core/synapse.svelte.ts'
import {
  allocateNetwork,
  releaseNetwork,
  addPopulationToNetwork,
  addSynapseGroupToNetwork,
  step,
  dopamine,
} from '../core/network.svelte.ts'
import {
  allocateSensor,
  releaseSensor,
  sendInput,
  sensoryRegistry,
} from '../core/sensory.svelte.ts'
import {
  allocateMotor,
  releaseMotor,
  decodeAction,
  motorRegistry,
} from '../core/motor.svelte.ts'

// ============================================================================
// WORM CONFIGURATION
// ============================================================================

/**
 * Population sizes (scaled from C. elegans ~300 neurons)
 * Scale 1.0 = ~300 neurons total
 * Scale 0.1 = ~30 neurons (for quick testing)
 */
interface WormConfig {
  // Sensory populations
  chemicalLeftSize: number   // Left chemical sensor (AWC-like)
  chemicalRightSize: number  // Right chemical sensor

  // Interneuron populations
  aibSize: number            // Turn promoter (AIB-like)
  aiySize: number            // Turn suppressor (AIY-like)

  // Motor populations
  forwardSize: number        // Forward command
  turnSize: number           // Turn command

  // Innate weights (these ARE the "DNA")
  // Calibrated so that pre_count × weight ≈ 10-15 (current needed to spike)
  chemicalToAib: number      // Excitatory: food → turn
  chemicalToAiy: number      // Inhibitory: food → suppress forward
  aibToTurn: number          // Excitatory: AIB → turn motor
  aiyToForward: number       // Excitatory: AIY → forward motor
  aibToAiy: number           // Inhibitory: mutual inhibition
  aiyToAib: number           // Inhibitory: mutual inhibition

  // Learning
  learningEnabled: boolean
  stdpRate: number
}

function getConfig(scale: number = 1.0): WormConfig {
  // Base sizes at scale 1.0 (~300 neurons total)
  const baseChemical = 20
  const baseInter = 50
  const baseMotor = 30

  const chemicalSize = Math.max(2, Math.round(baseChemical * scale))
  const interSize = Math.max(5, Math.round(baseInter * scale))
  const motorSize = Math.max(3, Math.round(baseMotor * scale))

  return {
    chemicalLeftSize: chemicalSize,
    chemicalRightSize: chemicalSize,
    aibSize: interSize,
    aiySize: interSize,
    forwardSize: motorSize,
    turnSize: motorSize,

    // Innate weights calibrated for chain transmission
    // Rule: pre_count × weight ≈ 10-15 (current to spike)
    // With chemicalSize=20, weight=0.6 → 20×0.6=12 ✓
    chemicalToAib: 0.6,      // Excitatory (positive)
    chemicalToAiy: -0.4,     // Inhibitory (negative) - weaker inhibition
    aibToTurn: 0.5,          // Excitatory
    aiyToForward: 0.5,       // Excitatory
    aibToAiy: -0.3,          // Mutual inhibition
    aiyToAib: -0.3,          // Mutual inhibition

    learningEnabled: true,
    stdpRate: 0.001,
  }
}

// ============================================================================
// WORM FACTORY
// ============================================================================

/**
 * Create a C. elegans-inspired worm creature.
 *
 * This factory IS the creature's "DNA" - the specific wiring encoded here
 * is what evolution discovered works for chemotaxis navigation.
 */
export function createWorm(id: string, options: CreatureOptions = {}): Creature {
  const scale = options.scale ?? 1.0
  const config = getConfig(scale)
  const learningEnabled = options.learningEnabled ?? config.learningEnabled

  // ============================================================================
  // ALLOCATE BRAIN STRUCTURE
  // ============================================================================

  // Network
  const networkIndex = allocateNetwork(`${id}_network`)

  // --- Sensory Populations ---
  const chemicalLeftPop = allocatePopulation(`${id}_chemical_left`, config.chemicalLeftSize, 'RS')
  const chemicalRightPop = allocatePopulation(`${id}_chemical_right`, config.chemicalRightSize, 'RS')

  // --- Interneuron Populations ---
  // AIB-like: turn promoter (excitatory type for output)
  const aibPop = allocatePopulation(`${id}_aib`, config.aibSize, 'RS')
  // AIY-like: turn suppressor (excitatory type for output)
  const aiyPop = allocatePopulation(`${id}_aiy`, config.aiySize, 'RS')

  // --- Motor Populations ---
  const forwardPop = allocatePopulation(`${id}_forward`, config.forwardSize, 'RS')
  const turnPop = allocatePopulation(`${id}_turn`, config.turnSize, 'RS')

  // Add all populations to network
  const allPops = [chemicalLeftPop, chemicalRightPop, aibPop, aiyPop, forwardPop, turnPop]
  for (const pop of allPops) {
    addPopulationToNetwork(networkIndex, pop)
  }

  // ============================================================================
  // CREATE INNATE WIRING (THE "DNA")
  // ============================================================================

  // Helper to create synapse group with proper connectivity
  function createConnection(
    name: string,
    prePop: number,
    postPop: number,
    weight: number,
    plastic: boolean = true
  ): number {
    const preSize = populationSize[prePop]
    const postSize = populationSize[postPop]
    const { preIndices, postIndices } = createAllToAllConnectivity(preSize, postSize)

    // Determine synapse type from weight sign
    const synapseType = weight >= 0 ? 'AMPA' : 'GABA_A'

    const groupIndex = allocateSynapseGroup(
      `${id}_${name}`,
      prePop,
      postPop,
      preIndices,
      postIndices,
      {
        synapseType,
        initialWeight: Math.abs(weight),
        // For inhibitory, weights should be negative
        minWeight: weight < 0 ? -1.0 : 0.0,
        maxWeight: weight < 0 ? 0.0 : 1.0,
        stdpEnabled: plastic && learningEnabled,
      }
    )

    return groupIndex
  }

  // --- Sensory → Interneuron connections ---
  // Chemical sensors excite AIB (turn promoter)
  const chemLeftToAib = createConnection('chem_left_aib', chemicalLeftPop, aibPop, config.chemicalToAib)
  const chemRightToAib = createConnection('chem_right_aib', chemicalRightPop, aibPop, config.chemicalToAib)

  // Chemical sensors inhibit AIY (suppress forward when food detected)
  const chemLeftToAiy = createConnection('chem_left_aiy', chemicalLeftPop, aiyPop, config.chemicalToAiy)
  const chemRightToAiy = createConnection('chem_right_aiy', chemicalRightPop, aiyPop, config.chemicalToAiy)

  // --- Interneuron → Motor connections ---
  // AIB excites turn motor
  const aibToTurn = createConnection('aib_turn', aibPop, turnPop, config.aibToTurn)

  // AIY excites forward motor
  const aiyToForward = createConnection('aiy_forward', aiyPop, forwardPop, config.aiyToForward)

  // --- Mutual inhibition between AIB and AIY ---
  // This creates winner-take-all competition
  const aibToAiy = createConnection('aib_aiy', aibPop, aiyPop, config.aibToAiy, false)  // Not plastic
  const aiyToAib = createConnection('aiy_aib', aiyPop, aibPop, config.aiyToAib, false)  // Not plastic

  // Add all synapse groups to network
  const allSynapses = [
    chemLeftToAib, chemRightToAib, chemLeftToAiy, chemRightToAiy,
    aibToTurn, aiyToForward, aibToAiy, aiyToAib
  ]
  for (const syn of allSynapses) {
    addSynapseGroupToNetwork(networkIndex, syn)
  }

  // ============================================================================
  // CREATE SENSORS AND MOTORS
  // ============================================================================

  // Sensors
  const chemicalLeftSensor = allocateSensor(`${id}_sensor_chem_left`, chemicalLeftPop, {
    type: 'chemical',
    encoding: 'rate',
    gain: 15.0,  // Amplify to produce spikes
  })

  const chemicalRightSensor = allocateSensor(`${id}_sensor_chem_right`, chemicalRightPop, {
    type: 'chemical',
    encoding: 'rate',
    gain: 15.0,
  })

  // Motors
  const forwardMotor = allocateMotor(`${id}_motor_forward`, forwardPop, {
    type: 'locomotion',
    decoding: 'rate',
    actionName: 'forward',
    gain: 1.0,
  })

  const turnMotor = allocateMotor(`${id}_motor_turn`, turnPop, {
    type: 'locomotion',
    decoding: 'rate',
    actionName: 'turn',
    gain: 1.0,
  })

  // ============================================================================
  // BUILD CREATURE INTERFACE
  // ============================================================================

  const populationIndices = allPops
  const synapseGroupIndices = allSynapses
  const sensorIndices = [chemicalLeftSensor, chemicalRightSensor]
  const motorIndices = [forwardMotor, turnMotor]

  // Named maps for semantic access
  const populations = new Map<string, number>([
    ['chemical_left', chemicalLeftPop],
    ['chemical_right', chemicalRightPop],
    ['aib', aibPop],
    ['aiy', aiyPop],
    ['forward', forwardPop],
    ['turn', turnPop],
  ])

  const sensors = new Map<string, number>([
    ['chemical_left', chemicalLeftSensor],
    ['chemical_right', chemicalRightSensor],
  ])

  const motors = new Map<string, number>([
    ['forward', forwardMotor],
    ['turn', turnMotor],
  ])

  // Track total reward
  let totalReward = 0

  // ============================================================================
  // CREATURE METHODS
  // ============================================================================

  function sense(stimuli: Map<string, number>): void {
    // Map stimuli to sensors
    const chemLeft = stimuli.get('chemical_left') ?? stimuli.get('chemical') ?? 0
    const chemRight = stimuli.get('chemical_right') ?? stimuli.get('chemical') ?? 0

    // Send to sensory populations
    sendInput(chemicalLeftSensor, chemLeft)
    sendInput(chemicalRightSensor, chemRight)
  }

  async function think(dt: number): Promise<void> {
    // Run one network step
    await step(networkIndex, dt)
  }

  function act(): Map<string, number> {
    const actions = new Map<string, number>()

    // Decode motor outputs
    const forwardAction = decodeAction(forwardMotor)
    const turnAction = decodeAction(turnMotor)

    actions.set('forward', forwardAction)
    actions.set('turn', turnAction)

    // Also provide turn_left and turn_right for world compatibility
    // Positive turn = left, negative = right
    actions.set('turn_left', Math.max(0, turnAction))
    actions.set('turn_right', Math.max(0, -turnAction))

    return actions
  }

  function setReward(value: number): void {
    totalReward += value
    // Set dopamine signal in network
    dopamine[networkIndex] = mx.array(value, mx.float32)
  }

  async function getState(): Promise<CreatureState> {
    // Collect spike counts from each population
    const populationActivity = new Map<string, number>()
    let totalSpikes = 0
    let activePopulations = 0

    for (const [name, popIndex] of populations) {
      const spikes = mx.sum(fired[popIndex])
      await mx.asyncEval(spikes)
      const count = spikes.item() as number
      populationActivity.set(name, count)
      totalSpikes += count
      if (count > 0) activePopulations++
    }

    // Get sensor values (last input)
    const sensorValues = new Map<string, number>()
    // Would need to track last input - for now just return 0
    sensorValues.set('chemical_left', 0)
    sensorValues.set('chemical_right', 0)

    // Get motor outputs
    const motorOutputs = act()

    // Get current reward
    await mx.asyncEval(dopamine[networkIndex])
    const currentReward = dopamine[networkIndex].item() as number

    return {
      id,
      totalSpikes,
      activePopulations,
      populationActivity,
      sensorValues,
      motorOutputs,
      currentReward,
      totalRewardReceived: totalReward,
    }
  }

  function destroy(): void {
    // Release all resources in reverse order
    for (const motorIndex of motorIndices) {
      releaseMotor(motorIndex)
    }
    for (const sensorIndex of sensorIndices) {
      releaseSensor(sensorIndex)
    }
    for (const synapseIndex of synapseGroupIndices) {
      releaseSynapseGroup(synapseIndex)
    }
    for (const popIndex of populationIndices) {
      releasePopulation(popIndex)
    }
    releaseNetwork(networkIndex)
  }

  // ============================================================================
  // RETURN CREATURE OBJECT
  // ============================================================================

  return {
    id,
    networkIndex,
    populationIndices,
    synapseGroupIndices,
    sensorIndices,
    motorIndices,
    populations,
    sensors,
    motors,
    sense,
    think,
    act,
    setReward,
    getState,
    destroy,
  }
}
