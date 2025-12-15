<script lang="ts">
/**
 * Live Navigation - Real-time terminal visualization of creature navigation
 *
 * Runs the simulation step-by-step with visual feedback.
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import WorldViz from './WorldViz.svelte'
import { createGradient } from '../worlds/gradient.svelte.ts'
import { createWorm } from '../creatures/worm.svelte.ts'
import { onMount, onDestroy } from 'svelte'

// State
let timestep = $state(0)
let creatureX = $state(25)
let creatureY = $state(25)
let creatureHeading = $state(0)
let forwardOutput = $state(0)
let turnOutput = $state(0)
let reward = $state(0)
let cumulativeReward = $state(0)
let running = $state(false)
let finished = $state(false)
let message = $state('Press Enter to start navigation simulation...')

// Configuration
const MAX_STEPS = 500
const STEP_DELAY = 50  // ms between steps

// World and creature
let world: ReturnType<typeof createGradient> | null = null
let creature: ReturnType<typeof createWorm> | null = null
let runInterval: ReturnType<typeof setInterval> | null = null

// Setup
function setup() {
  world = createGradient('viz_world', {
    width: 100,
    height: 100,
    sourceX: 75,
    sourceY: 75,
    decayRate: 0.03,
    rewardRadius: 10,
  })

  creature = createWorm('viz_creature', {
    scale: 0.5,
    learningEnabled: true,  // Enable STDP
  })

  world.addCreature(creature, { x: 25, y: 25, heading: 0 })

  // Reset state
  timestep = 0
  creatureX = 25
  creatureY = 25
  creatureHeading = 0
  forwardOutput = 0
  turnOutput = 0
  reward = 0
  cumulativeReward = 0
  finished = false
}

// Single step
async function simulationStep() {
  if (!world || !creature || finished) return

  // Sense
  const stimuli = world.getStimuli(creature)
  creature.sense(stimuli)

  // Think
  await creature.think(1.0)

  // Act
  const actions = creature.act()
  world.applyActions(creature, actions)

  // Get reward
  const r = world.getReward(creature)
  creature.setReward(r)

  // Update state
  const pos = world.getPosition(creature)
  if (pos) {
    creatureX = pos.x
    creatureY = pos.y
    creatureHeading = pos.heading ?? 0
  }

  forwardOutput = actions.get('forward') ?? 0
  turnOutput = actions.get('turn') ?? 0
  reward = r
  cumulativeReward += r

  // World step
  world.step(1.0)

  timestep++

  // Check if reached target or max steps
  const dx = creatureX - 75
  const dy = creatureY - 75
  const distToTarget = Math.sqrt(dx*dx + dy*dy)

  if (distToTarget < 10) {
    message = `SUCCESS! Reached food source in ${timestep} steps! Total reward: ${cumulativeReward.toFixed(2)}`
    stop()
    finished = true
  } else if (timestep >= MAX_STEPS) {
    message = `Finished ${MAX_STEPS} steps. Final distance to target: ${distToTarget.toFixed(1)}. Total reward: ${cumulativeReward.toFixed(2)}`
    stop()
    finished = true
  }
}

// Start simulation
function start() {
  if (running) return
  if (finished) {
    cleanup()
    setup()
  }

  running = true
  message = 'Running navigation simulation...'

  runInterval = setInterval(async () => {
    await simulationStep()
  }, STEP_DELAY)
}

// Stop simulation
function stop() {
  running = false
  if (runInterval) {
    clearInterval(runInterval)
    runInterval = null
  }
  if (!finished) {
    message = `Paused at step ${timestep}. Press Enter to continue, R to restart.`
  }
}

// Cleanup
function cleanup() {
  stop()
  creature?.destroy()
  world?.destroy()
  creature = null
  world = null
}

// Handle input
function handleInput(key: string) {
  if (key === 'q') {
    cleanup()
    process.exit(0)
  } else if (key === 'r' || key === 'R') {
    cleanup()
    setup()
    message = 'Reset. Press Enter to start navigation simulation...'
  } else if (key === '\r' || key === ' ') {
    if (running) {
      stop()
    } else {
      start()
    }
  }
}

onMount(() => {
  setup()

  // Handle keyboard input
  process.stdin.setRawMode?.(true)
  process.stdin.on('data', (data) => {
    const key = data.toString()
    handleInput(key)
  })
})

onDestroy(() => {
  cleanup()
})
</script>

<box flexDirection="column" padding={1}>
  <text bold color="green">VESSEL - Live Navigation</text>
  <text color="gray">────────────────────────────────────────────────</text>

  <WorldViz
    width={50}
    height={25}
    worldWidth={100}
    worldHeight={100}
    {creatureX}
    {creatureY}
    {creatureHeading}
    sourceX={75}
    sourceY={75}
    {forwardOutput}
    {turnOutput}
    {reward}
    {timestep}
    {cumulativeReward}
  />

  <text color="yellow">{message}</text>
  <text color="gray">Controls: [Enter/Space] Start/Pause | [R] Restart | [Q] Quit</text>
</box>
