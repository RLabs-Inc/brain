/**
 * Live Navigation Visualization
 *
 * Real-time terminal display of creature navigating toward food.
 * Uses simple console output with ANSI escape codes for animation.
 *
 * Usage: bun run dist/src/suites/navigation/live.mjs
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { createGradient } from '../../worlds/gradient.svelte.ts'
import { createWorm } from '../../creatures/worm.svelte.ts'

// Configuration
const MAX_STEPS = 500
const STEP_DELAY = 80  // ms between steps
const DISPLAY_WIDTH = 60
const DISPLAY_HEIGHT = 30
const WORLD_WIDTH = 100
const WORLD_HEIGHT = 100
const SOURCE_X = 75
const SOURCE_Y = 75

// ANSI escape codes
const CLEAR = '\x1b[2J\x1b[H'
const BOLD = '\x1b[1m'
const RESET = '\x1b[0m'
const GREEN = '\x1b[32m'
const YELLOW = '\x1b[33m'
const CYAN = '\x1b[36m'
const MAGENTA = '\x1b[35m'
const RED = '\x1b[31m'
const GRAY = '\x1b[90m'

// Scale world coordinates to display coordinates
function toDisplayX(x: number): number {
  return Math.round((x / WORLD_WIDTH) * (DISPLAY_WIDTH - 1))
}

function toDisplayY(y: number): number {
  return Math.round((y / WORLD_HEIGHT) * (DISPLAY_HEIGHT - 1))
}

// Get heading character
function getHeadingChar(heading: number): string {
  const h = ((heading % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)

  if (h < Math.PI / 8 || h >= 15 * Math.PI / 8) return '→'
  if (h < 3 * Math.PI / 8) return '↗'
  if (h < 5 * Math.PI / 8) return '↑'
  if (h < 7 * Math.PI / 8) return '↖'
  if (h < 9 * Math.PI / 8) return '←'
  if (h < 11 * Math.PI / 8) return '↙'
  if (h < 13 * Math.PI / 8) return '↓'
  return '↘'
}

// Build display frame
function renderFrame(
  creatureX: number,
  creatureY: number,
  heading: number,
  timestep: number,
  forwardOutput: number,
  turnOutput: number,
  cumulativeReward: number,
  message: string
): string {
  const lines: string[] = []

  // Header
  lines.push(`${BOLD}${GREEN}VESSEL - Live Navigation${RESET}`)
  lines.push(`${GRAY}${'─'.repeat(DISPLAY_WIDTH + 2)}${RESET}`)

  // Display coordinates
  const displayCreatureX = toDisplayX(creatureX)
  const displayCreatureY = toDisplayY(creatureY)
  const displaySourceX = toDisplayX(SOURCE_X)
  const displaySourceY = toDisplayY(SOURCE_Y)

  // Grid
  lines.push('┌' + '─'.repeat(DISPLAY_WIDTH) + '┐')

  for (let y = 0; y < DISPLAY_HEIGHT; y++) {
    let line = '│'

    for (let x = 0; x < DISPLAY_WIDTH; x++) {
      if (x === displayCreatureX && y === displayCreatureY) {
        // Creature with heading
        line += `${CYAN}${getHeadingChar(heading)}${RESET}`
      } else if (x === displaySourceX && y === displaySourceY) {
        // Food source
        line += `${RED}◎${RESET}`
      } else if (Math.abs(x - displaySourceX) <= 3 && Math.abs(y - displaySourceY) <= 3) {
        // Near food (reward radius)
        line += `${GRAY}·${RESET}`
      } else {
        line += ' '
      }
    }

    line += '│'
    lines.push(line)
  }

  lines.push('└' + '─'.repeat(DISPLAY_WIDTH) + '┘')

  // Stats
  lines.push('')
  lines.push(`${YELLOW}t=${timestep.toString().padStart(4)}${RESET} | ` +
    `pos=(${creatureX.toFixed(1).padStart(5)}, ${creatureY.toFixed(1).padStart(5)}) | ` +
    `heading: ${(heading * 180 / Math.PI).toFixed(0).padStart(4)}°`)

  lines.push(`${GREEN}forward=${forwardOutput.toFixed(2).padStart(6)}${RESET} | ` +
    `${MAGENTA}turn=${turnOutput.toFixed(2).padStart(6)}${RESET} | ` +
    `${CYAN}reward=${cumulativeReward.toFixed(2)}${RESET}`)

  // Distance to target
  const dx = creatureX - SOURCE_X
  const dy = creatureY - SOURCE_Y
  const dist = Math.sqrt(dx*dx + dy*dy)
  const progress = Math.max(0, 100 - dist).toFixed(1)
  lines.push(`Distance to food: ${dist.toFixed(1)} | Progress: ${progress}%`)

  // Message
  lines.push('')
  lines.push(message)

  // Legend
  lines.push(`${GRAY}${RED}◎${GRAY} = food source | ${CYAN}${getHeadingChar(heading)}${GRAY} = creature${RESET}`)

  return lines.join('\n')
}

// Sleep helper
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// Main simulation
async function main() {
  console.log('Creating world and creature...')

  const world = createGradient('live_world', {
    width: WORLD_WIDTH,
    height: WORLD_HEIGHT,
    sourceX: SOURCE_X,
    sourceY: SOURCE_Y,
    decayRate: 0.03,
    rewardRadius: 10,
  })

  const creature = createWorm('live_creature', {
    scale: 0.5,
    learningEnabled: true,
  })

  world.addCreature(creature, { x: 25, y: 25, heading: 0 })

  let timestep = 0
  let cumulativeReward = 0
  let message = 'Navigating...'
  let finished = false

  // Run simulation
  while (timestep < MAX_STEPS && !finished) {
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
    cumulativeReward += r

    // Get position
    const pos = world.getPosition(creature)
    const creatureX = pos?.x ?? 25
    const creatureY = pos?.y ?? 25
    const heading = pos?.heading ?? 0
    const forwardOutput = actions.get('forward') ?? 0
    const turnOutput = actions.get('turn') ?? 0

    // World step
    world.step(1.0)
    timestep++

    // Check if reached target
    const dx = creatureX - SOURCE_X
    const dy = creatureY - SOURCE_Y
    const distToTarget = Math.sqrt(dx*dx + dy*dy)

    if (distToTarget < 10) {
      message = `${GREEN}${BOLD}SUCCESS! Reached food source in ${timestep} steps!${RESET}`
      finished = true
    } else if (timestep >= MAX_STEPS) {
      message = `Finished ${MAX_STEPS} steps. Distance: ${distToTarget.toFixed(1)}`
      finished = true
    }

    // Render frame
    const frame = renderFrame(
      creatureX,
      creatureY,
      heading,
      timestep,
      forwardOutput,
      turnOutput,
      cumulativeReward,
      message
    )

    process.stdout.write(CLEAR + frame)

    // Wait
    await sleep(STEP_DELAY)
  }

  // Final stats
  console.log('')
  console.log(`${BOLD}Final Results:${RESET}`)
  console.log(`  Steps: ${timestep}`)
  console.log(`  Total Reward: ${cumulativeReward.toFixed(2)}`)

  const finalPos = world.getPosition(creature)
  const finalDist = Math.sqrt(
    Math.pow((finalPos?.x ?? 0) - SOURCE_X, 2) +
    Math.pow((finalPos?.y ?? 0) - SOURCE_Y, 2)
  )
  console.log(`  Final Distance to Food: ${finalDist.toFixed(1)}`)

  // Cleanup
  creature.destroy()
  world.destroy()
}

main().catch(console.error)
