/**
 * Navigation Experiment Helpers
 *
 * Shared utilities for navigation experiments.
 * Includes variant creature factories for different experiment conditions.
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import type { Creature, CreatureState, CreatureOptions } from '../../creatures/types.ts'
import {
  allocatePopulation,
  releasePopulation,
  injectCurrent,
  voltage,
  fired,
  populationSize,
} from '../../core/neuron.svelte.ts'
import {
  allocateSynapseGroup,
  releaseSynapseGroup,
  createAllToAllConnectivity,
  weights,
} from '../../core/synapse.svelte.ts'
import {
  allocateNetwork,
  releaseNetwork,
  addPopulationToNetwork,
  addSynapseGroupToNetwork,
  step,
  dopamine,
} from '../../core/network.svelte.ts'
import {
  allocateSensor,
  releaseSensor,
  sendInput,
} from '../../core/sensory.svelte.ts'
import {
  allocateMotor,
  releaseMotor,
  decodeAction,
  updateSpikeWindow,
} from '../../core/motor.svelte.ts'

// ============================================================================
// SHARED CONFIG
// ============================================================================

function getBaseConfig(scale: number) {
  const chemicalSize = Math.max(2, Math.round(20 * scale))
  const interSize = Math.max(5, Math.round(50 * scale))
  const motorSize = Math.max(3, Math.round(30 * scale))

  return { chemicalSize, interSize, motorSize }
}

// ============================================================================
// RANDOM WORM - For baseline comparison
// ============================================================================

/**
 * Create a worm with RANDOM wiring.
 * Same architecture as innate worm, but random weights.
 * This is the control - if this navigates, the task is too easy.
 */
export function createRandomWorm(id: string, options: CreatureOptions = {}): Creature {
  const scale = options.scale ?? 1.0
  const { chemicalSize, interSize, motorSize } = getBaseConfig(scale)
  const seed = options.seed ?? Date.now()

  // Seed random (simple LCG for reproducibility)
  let rng = seed
  const random = () => {
    rng = (rng * 1103515245 + 12345) & 0x7fffffff
    return rng / 0x7fffffff
  }

  // Network
  const networkIndex = allocateNetwork(`${id}_network`)

  // Populations (same structure as innate worm)
  const chemicalLeftPop = allocatePopulation(`${id}_chem_left`, chemicalSize, 'RS')
  const chemicalRightPop = allocatePopulation(`${id}_chem_right`, chemicalSize, 'RS')
  const aibPop = allocatePopulation(`${id}_aib`, interSize, 'RS')
  const aiyPop = allocatePopulation(`${id}_aiy`, interSize, 'RS')
  const forwardPop = allocatePopulation(`${id}_forward`, motorSize, 'RS')
  const turnPop = allocatePopulation(`${id}_turn`, motorSize, 'RS')

  const allPops = [chemicalLeftPop, chemicalRightPop, aibPop, aiyPop, forwardPop, turnPop]
  for (const pop of allPops) {
    addPopulationToNetwork(networkIndex, pop)
  }

  // Create connections with RANDOM weights (±0.5)
  function createRandomConnection(name: string, pre: number, post: number): number {
    const preSize = populationSize[pre]
    const postSize = populationSize[post]
    const { preIndices, postIndices } = createAllToAllConnectivity(preSize, postSize)

    // Random weight: -0.5 to +0.5
    const weight = (random() - 0.5)
    const synapseType = weight >= 0 ? 'AMPA' : 'GABA_A'

    return allocateSynapseGroup(
      `${id}_${name}`,
      pre,
      post,
      preIndices,
      postIndices,
      {
        synapseType,
        initialWeight: Math.abs(weight),
        minWeight: weight < 0 ? -1.0 : 0.0,
        maxWeight: weight < 0 ? 0.0 : 1.0,
        stdpEnabled: false,  // No learning in random baseline
      }
    )
  }

  // Same connectivity PATTERN as innate, but RANDOM weights
  const allSynapses = [
    createRandomConnection('chem_left_aib', chemicalLeftPop, aibPop),
    createRandomConnection('chem_right_aib', chemicalRightPop, aibPop),
    createRandomConnection('chem_left_aiy', chemicalLeftPop, aiyPop),
    createRandomConnection('chem_right_aiy', chemicalRightPop, aiyPop),
    createRandomConnection('aib_turn', aibPop, turnPop),
    createRandomConnection('aiy_forward', aiyPop, forwardPop),
    createRandomConnection('aib_aiy', aibPop, aiyPop),
    createRandomConnection('aiy_aib', aiyPop, aibPop),
  ]

  for (const syn of allSynapses) {
    addSynapseGroupToNetwork(networkIndex, syn)
  }

  // Sensors
  const chemicalLeftSensor = allocateSensor(`${id}_sensor_left`, chemicalLeftPop, {
    type: 'chemical',
    encoding: 'rate',
    gain: 15.0,
  })
  const chemicalRightSensor = allocateSensor(`${id}_sensor_right`, chemicalRightPop, {
    type: 'chemical',
    encoding: 'rate',
  })

  // Motors - same gains as innate worm for fair comparison
  const forwardMotor = allocateMotor(`${id}_motor_forward`, forwardPop, {
    type: 'locomotion',
    decoding: 'rate',
    actionName: 'forward',
    gain: 50.0,
    smoothing: 0.3,
  })
  const turnMotor = allocateMotor(`${id}_motor_turn`, turnPop, {
    type: 'locomotion',
    decoding: 'rate',
    actionName: 'turn',
    gain: 20.0,
    smoothing: 0.3,
  })

  // Build creature interface
  const sensorIndices = [chemicalLeftSensor, chemicalRightSensor]
  const motorIndices = [forwardMotor, turnMotor]

  const populations = new Map([
    ['chemical_left', chemicalLeftPop],
    ['chemical_right', chemicalRightPop],
    ['aib', aibPop],
    ['aiy', aiyPop],
    ['forward', forwardPop],
    ['turn', turnPop],
  ])

  const sensors = new Map([
    ['chemical_left', chemicalLeftSensor],
    ['chemical_right', chemicalRightSensor],
  ])

  const motors = new Map([
    ['forward', forwardMotor],
    ['turn', turnMotor],
  ])

  // ============================================================================
  // CACHED TONIC DRIVE ARRAYS (avoid creating new Metal buffers every think()!)
  // NOTE: We can cache DRIVE values but NOT indices (scatter-add corrupts them)
  // ============================================================================
  const aiySize = populationSize[aiyPop]
  const aibSize = populationSize[aibPop]
  const cachedAiyDrive = mx.full([aiySize], 12.0, mx.float32)
  const cachedAibDrive = mx.full([aibSize], 3.0, mx.float32)
  // NO cached indices - scatter-add corrupts them!

  let totalReward = 0

  function sense(stimuli: Map<string, number>): void {
    const chemLeft = stimuli.get('chemical_left') ?? stimuli.get('chemical') ?? 0
    const chemRight = stimuli.get('chemical_right') ?? stimuli.get('chemical') ?? 0
    sendInput(chemicalLeftSensor, chemLeft)
    sendInput(chemicalRightSensor, chemRight)
  }

  async function think(dt: number): Promise<void> {
    // TONIC ACTIVITY: Same drive as innate worm for fair comparison
    // The only difference should be the WIRING, not the drive to move
    // NOTE: Drive values cached, but indices must be fresh (scatter-add bug)

    // AIY gets strong tonic drive (moves forward by default)
    injectCurrent(aiyPop, mx.arange(0, aiySize, 1, mx.int32), cachedAiyDrive)

    // AIB gets weak tonic drive (occasional spontaneous turns)
    injectCurrent(aibPop, mx.arange(0, aibSize, 1, mx.int32), cachedAibDrive)

    await step(networkIndex, dt)

    // Update motor spike windows (needed for rate decoding)
    for (const m of motorIndices) {
      updateSpikeWindow(m)
    }
  }

  function act(): Map<string, number> {
    const actions = new Map<string, number>()
    actions.set('forward', decodeAction(forwardMotor))
    actions.set('turn', decodeAction(turnMotor))
    actions.set('turn_left', Math.max(0, actions.get('turn')!))
    actions.set('turn_right', Math.max(0, -(actions.get('turn')!)))
    return actions
  }

  function setReward(value: number): void {
    totalReward += value
    dopamine[networkIndex] = mx.array(value, mx.float32)
  }

  async function getState(): Promise<CreatureState> {
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

    return {
      id,
      totalSpikes,
      activePopulations,
      populationActivity,
      sensorValues: new Map(),
      motorOutputs: act(),
      currentReward: (dopamine[networkIndex]?.item() as number) ?? 0,
      totalRewardReceived: totalReward,
    }
  }

  function destroy(): void {
    for (const m of motorIndices) releaseMotor(m)
    for (const s of sensorIndices) releaseSensor(s)
    for (const syn of allSynapses) releaseSynapseGroup(syn)
    for (const pop of allPops) releasePopulation(pop)
    releaseNetwork(networkIndex)
  }

  return {
    id,
    networkIndex,
    populationIndices: allPops,
    synapseGroupIndices: allSynapses,
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

// ============================================================================
// INNATE WORM - Re-export from creatures module
// ============================================================================

export { createWorm as createInnateWorm } from '../../creatures/worm.svelte.ts'

// ============================================================================
// LEARNING WORM - Innate wiring WITH STDP enabled
// ============================================================================

/**
 * Create a worm with innate wiring AND learning enabled.
 * This is the test condition - can it improve on innate alone?
 */
export function createLearningWorm(id: string, options: CreatureOptions = {}): Creature {
  // Just use the regular worm factory with learning enabled
  const { createWorm } = require('../../creatures/worm.svelte.ts')
  return createWorm(id, { ...options, learningEnabled: true })
}
