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
  updateSpikeWindow,
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

  // KLINOKINESIS sensors - temporal comparison "am I getting warmer?"
  // This is THE KEY to real chemotaxis!
  const gettingWarmerPop = allocatePopulation(`${id}_getting_warmer`, config.chemicalLeftSize, 'RS')
  const gettingColderPop = allocatePopulation(`${id}_getting_colder`, config.chemicalLeftSize, 'RS')

  // --- Interneuron Populations ---
  // AIB-like: turn promoter (excitatory type for output)
  const aibPop = allocatePopulation(`${id}_aib`, config.aibSize, 'RS')
  // AIY-like: turn suppressor (excitatory type for output)
  const aiyPop = allocatePopulation(`${id}_aiy`, config.aiySize, 'RS')

  // --- Motor Populations ---
  const forwardPop = allocatePopulation(`${id}_forward`, config.forwardSize, 'RS')
  const turnPop = allocatePopulation(`${id}_turn`, config.turnSize, 'RS')

  // Add all populations to network
  const allPops = [
    chemicalLeftPop, chemicalRightPop,
    gettingWarmerPop, gettingColderPop,  // Klinokinesis sensors
    aibPop, aiyPop,
    forwardPop, turnPop
  ]
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
  // Chemical sensors provide weak local search (secondary to klinokinesis)
  const chemLeftToAib = createConnection('chem_left_aib', chemicalLeftPop, aibPop, config.chemicalToAib * 0.3)  // Weakened
  const chemRightToAib = createConnection('chem_right_aib', chemicalRightPop, aibPop, config.chemicalToAib * 0.3)

  // KLINOKINESIS - THE KEY TO REAL CHEMOTAXIS!
  // "Getting colder" (concentration decreasing) → TURN MORE
  // This is the biological algorithm: if you're going the wrong way, change direction!
  const colderToAib = createConnection('colder_to_aib', gettingColderPop, aibPop, 0.8)  // Strong: turn when wrong!

  // "Getting warmer" (concentration increasing) → GO STRAIGHT
  // If you're going the right way, keep going!
  const warmerToAiy = createConnection('warmer_to_aiy', gettingWarmerPop, aiyPop, 0.8)  // Strong: forward when right!

  // Chemical sensors inhibit AIY (suppress forward when food detected - local search)
  const chemLeftToAiy = createConnection('chem_left_aiy', chemicalLeftPop, aiyPop, config.chemicalToAiy * 0.3)  // Weakened
  const chemRightToAiy = createConnection('chem_right_aiy', chemicalRightPop, aiyPop, config.chemicalToAiy * 0.3)

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
    colderToAib, warmerToAiy,  // KLINOKINESIS connections
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

  // KLINOKINESIS SENSORS - the key to real chemotaxis!
  const gettingWarmerSensor = allocateSensor(`${id}_sensor_warmer`, gettingWarmerPop, {
    type: 'chemical',
    encoding: 'rate',
    gain: 20.0,  // Strong response to temporal changes
  })

  const gettingColderSensor = allocateSensor(`${id}_sensor_colder`, gettingColderPop, {
    type: 'chemical',
    encoding: 'rate',
    gain: 20.0,  // Strong response to temporal changes
  })

  // Motors
  // Gain increased and smoothing reduced for responsive movement
  // With rate coding over 20-step window, raw output is ~0.01-0.05
  // Need gain ~50-100 to produce meaningful movement (0.5-5 units/step)
  const forwardMotor = allocateMotor(`${id}_motor_forward`, forwardPop, {
    type: 'locomotion',
    decoding: 'rate',
    actionName: 'forward',
    gain: 50.0,       // Amplify small rate outputs to meaningful movement
    smoothing: 0.3,   // Faster response (70% of new signal, 30% of old)
  })

  const turnMotor = allocateMotor(`${id}_motor_turn`, turnPop, {
    type: 'locomotion',
    decoding: 'rate',
    actionName: 'turn',
    gain: 20.0,       // Turning doesn't need as much gain
    smoothing: 0.3,
  })

  // ============================================================================
  // BUILD CREATURE INTERFACE
  // ============================================================================

  const populationIndices = allPops
  const synapseGroupIndices = allSynapses
  const sensorIndices = [chemicalLeftSensor, chemicalRightSensor, gettingWarmerSensor, gettingColderSensor]
  const motorIndices = [forwardMotor, turnMotor]

  // Named maps for semantic access
  const populations = new Map<string, number>([
    ['chemical_left', chemicalLeftPop],
    ['chemical_right', chemicalRightPop],
    ['getting_warmer', gettingWarmerPop],
    ['getting_colder', gettingColderPop],
    ['aib', aibPop],
    ['aiy', aiyPop],
    ['forward', forwardPop],
    ['turn', turnPop],
  ])

  // ============================================================================
  // CACHED TONIC DRIVE ARRAYS (avoid creating new Metal buffers every think()!)
  // These are the "primal drive" currents - created once, reused forever
  // NOTE: We can cache the DRIVE values but NOT the indices!
  // The scatter-add bug (array.at(indices).add(values)) corrupts index arrays
  // after first use, so indices must be created fresh each call.
  // ============================================================================
  const aiySize = populationSize[aiyPop]
  const aibSize = populationSize[aibPop]
  const cachedAiyDrive = mx.full([aiySize], 12.0, mx.float32)  // Strong tonic drive
  const cachedAibDrive = mx.full([aibSize], 3.0, mx.float32)   // Weak tonic drive
  // NO cached indices - scatter-add corrupts them!

  const sensors = new Map<string, number>([
    ['chemical_left', chemicalLeftSensor],
    ['chemical_right', chemicalRightSensor],
    ['getting_warmer', gettingWarmerSensor],
    ['getting_colder', gettingColderSensor],
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

    // KLINOKINESIS - temporal comparison signals
    // These are THE KEY to real chemotaxis!
    const gettingWarmer = stimuli.get('concentration_increasing') ?? 0
    const gettingColder = stimuli.get('concentration_decreasing') ?? 0

    // Send to sensory populations
    sendInput(chemicalLeftSensor, chemLeft)
    sendInput(chemicalRightSensor, chemRight)
    sendInput(gettingWarmerSensor, gettingWarmer)
    sendInput(gettingColderSensor, gettingColder)
  }

  async function think(dt: number): Promise<void> {
    // TONIC ACTIVITY: The "primal drive to explore"
    // In biology, AIY has baseline firing that drives forward movement.
    // This represents the innate survival instinct - always be moving, searching.
    // Without this, the worm would have no motivation to move at all!
    //
    // Inject constant current to AIY (forward drive) and small amount to AIB
    // This creates the baseline behavior: move forward, occasionally turn
    //
    // NOTE: Drive VALUES are cached, but INDICES must be created fresh each call
    // because scatter-add (array.at(indices).add(values)) corrupts index arrays!

    // AIY gets strong tonic drive (moves forward by default)
    injectCurrent(aiyPop, mx.arange(0, aiySize, 1, mx.int32), cachedAiyDrive)

    // AIB gets weak tonic drive (occasional spontaneous turns)
    injectCurrent(aibPop, mx.arange(0, aibSize, 1, mx.int32), cachedAibDrive)

    // Run one network step
    await step(networkIndex, dt)

    // Update motor spike windows (needed for rate decoding)
    for (const motorIndex of motorIndices) {
      updateSpikeWindow(motorIndex)
    }
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
